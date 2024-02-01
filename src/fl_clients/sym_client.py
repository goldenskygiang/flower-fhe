from collections import OrderedDict
from logging import INFO

import torch
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code)

from crypto.rsa_crypto import RsaCryptoAPI
from models import get_model, train, test

class SymClient(fl.client.Client):
    def __init__(self, cid, dl_train, dl_val, device=None) -> None:
        super().__init__()
        self.cid = cid
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.device = device
        self.model = get_model()

        if device:
            self.model = self.model.to(device)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        '''
        Extract all model's params and convert to a list of
        NumPy arrays, then encrypt. Server doesn't work with PyTorch, TF...
        '''
        enc_params = [RsaCryptoAPI.encrypt_numpy_array(ins.config['aes_key'], val.cpu().numpy()) \
                      for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=enc_params, tensor_type="")
        )

    def set_parameters(self, parameters: Parameters, aes_key: bytes):
        '''
        With the model's params received from central server,
        decrypt them and overwrite the unintialized model in this class
        '''
        # get params into a dict
        params_dict = zip(self.model.state_dict().keys(), parameters.tensors,
                          [v.shape for k, v in self.model.state_dict().items()],
                          [v.dtype for k, v in self.model.state_dict().items()])

        state_dict = OrderedDict({
            k: torch.frombuffer(RsaCryptoAPI.decrypt_obj(aes_key, v), dtype=dtype).reshape(shape) \
            for k, v, shape, dtype in params_dict
        })
        # replace params
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, ins: FitIns) -> FitRes:
        '''
        Trains the model using the params sent by server, on
        this client's dataset. At the end, the params (locally
        trained) are comminucated back to the server)
        '''
        log(INFO, f'Client {self.cid} training')

        # copy params from server
        aes_key = RsaCryptoAPI.decrypt_aes_key(ins.config['private_key_pem'], ins.config['enc_key'])
        self.set_parameters(ins.parameters, aes_key)

        # define optimizer
        #optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optim = torch.optim.SGD([
            {'params': list(self.model.parameters())[:-1], 'lr': 1e-4, 'momentum':0.9},
            {'params': list(self.model.parameters())[-1], 'lr': 5e-2, 'momentum': 0.9}
        ])

        # local training
        trained_model, hist = train(self.model, self.dl_train, optim, epochs=1, device=self.device)

        # return model's params to the server, as well as extra info (number of training samples)
        get_param_ins = GetParametersIns(config={
            'aes_key': RsaCryptoAPI.decrypt_aes_key(ins.config['private_key_pem'], ins.config['enc_key'])
        })

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=self.get_parameters(get_param_ins).parameters,
            num_examples=len(self.dl_train),
            metrics=hist
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        '''
        Evaluate the model sent by server on the local client's
        local validation set. Returns performance metrics
        '''
        log(INFO, f'Client {self.cid} evaluating')

        aes_key = RsaCryptoAPI.decrypt_aes_key(ins.config['private_key_pem'], ins.config['enc_key'])
        self.set_parameters(ins.parameters, aes_key)

        loss, metrics_dict = test(self.model, self.dl_val, device=self.device)

        # send back to server
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.dl_val),
            metrics=metrics_dict
        )