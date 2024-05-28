from collections import OrderedDict
from logging import INFO, WARNING

import random as rd
from typing import Callable
from numpy import ndarray
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

from crypto.fhe_crypto import FheCryptoAPI
from models import train, test

import pickle

class FheClient(fl.client.Client):
    def __init__(
            self, cid, dl_train, dl_val, init_model_fn: Callable, device=None,
            straggler_sched: list[int]=[], proximal_mu: float=0, epochs: int=1) -> None:
        log(INFO, f"FHE client {cid} created")

        super().__init__()
        self.cid = cid
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.device = device
        self.model = init_model_fn()
        self.straggler_sched = straggler_sched
        self.proximal_mu = proximal_mu
        self.epochs = epochs

        if device:
            self.model = self.model.to(device)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        '''
        Extract all model's params and convert to a list of
        NumPy arrays, then encrypt. Server doesn't work with PyTorch, TF...
        '''
        log(INFO, "Client GETting params")
        cc = ins.config['crypto_context']
        pubkey = ins.config['public_key']
        
        enc_params = [
            pickle.dumps(FheCryptoAPI.encrypt_numpy_array(cc, pubkey, val.cpu().numpy())) \
            for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=enc_params, tensor_type="")
        )

    def set_parameters(self, parameters: Parameters, config: dict):
        '''
        With the model's params received from central server,
        decrypt them and overwrite the unintialized model in this class
        '''
        log(INFO, "Client setting params")

        cc = config['crypto_context']
        seckey = config['secret_key']

        # get params into a dict
        params_dict = zip(self.model.state_dict().keys(),
                          parameters.tensors,
                          [v.shape for k, v in self.model.state_dict().items()],
                          [v.dtype for k, v in self.model.state_dict().items()])

        state_dict = OrderedDict({
            k: FheCryptoAPI.decrypt_torch_tensor(cc, seckey, pickle.loads(tensor), dtype, shape) \
            for k, tensor, shape, dtype in params_dict
        })
        # replace params
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, ins: FitIns) -> FitRes:
        '''
        Trains the model using the params sent by server, on
        this client's dataset. At the end, the params (locally
        trained) are comminucated back to the server)
        '''
        log(INFO, "FITTING")
        log(INFO, f"Start training round {ins.config['curr_round']}")

        is_straggler = 0
        if ins.config['curr_round'] > 0 and len(self.straggler_sched) > 0:
            sv_round = (int(ins.config['curr_round']) - 1) % len(self.straggler_sched)
            is_straggler = self.straggler_sched[sv_round]

        if is_straggler == 0:
            # copy params from server
            self.set_parameters(ins.parameters, ins.config)

            log(INFO, f'Client {self.cid} training')
            # define optimizer
            #optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            optim = torch.optim.SGD([
                {'params': list(self.model.parameters())[:-1], 'lr': 1e-4, 'momentum':0.9},
                {'params': list(self.model.parameters())[-1], 'lr': 5e-2, 'momentum': 0.9}
            ])

            # local training
            train(ins.config['ds'], self.model, self.dl_train, optim, epochs=self.epochs,
                  device=self.device, proximal_mu=self.proximal_mu)
            
            # return model's params to the server, as well as extra info (number of training samples)
            get_param_ins = GetParametersIns(config={
                'crypto_context': ins.config['crypto_context'],
                'public_key': ins.config['public_key']
            })

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=self.get_parameters(get_param_ins).parameters,
                num_examples=len(self.dl_train),
                metrics={ "is_straggler": is_straggler }
            )
        else:
            log(WARNING, f"Client {self.cid} is a straggler in round {ins.config['curr_round']}")
            return FitRes(
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Straggler"),
                parameters=Parameters([], ''),
                num_examples=len(self.dl_train),
                metrics={ "is_straggler": is_straggler }
            )
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        '''
        Evaluate the model sent by server on the local client's
        local validation set. Returns performance metrics
        '''
        if ins.config['skip']:
            log(WARNING, f'Client {self.cid} skip evaluation this round')
            loss = 0.0
            accuracy = 0.0
        else:
            log(INFO, f'Client {self.cid} evaluating')
            self.set_parameters(ins.parameters, ins.config)
            loss, accuracy = test(ins.config['ds'], self.model, self.dl_val, device=self.device)
            loss = float(loss)

        # send back to server
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=len(self.dl_val),
            metrics={'accuracy': accuracy}
        )