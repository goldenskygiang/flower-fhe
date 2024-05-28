import pickle
import time
from crypto.rsa_crypto import RsaCryptoAPI

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from logging import INFO, WARNING

import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    NDArrays,
    Scalar,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    MetricsAggregationFn)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class SymFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        init_model_fn: Callable,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        dataset_name: str = None
    ) -> None:
        """FedAvg strategy with symmmetric encryption

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.model = init_model_fn()
        self.init_stage = True
        self.dataset_name = dataset_name

        self.__aes_key = RsaCryptoAPI.gen_aes_key()
        self.ckpt_name = ""

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    def initialize_parameters(self, client_manager: ClientManager):
        #log(INFO, f'Server AES key: {self.__aes_key}')
        # TODO: Save initial checkpoint
        return Parameters(
            tensors=[v.cpu().numpy() for _, v in self.model.state_dict().items()],
            tensor_type="numpy.ndarrays")
    
    def _get_param_info(self):
        return zip([],
                    [v.shape for k, v in self.model.state_dict().items()],
                    [v.dtype for k, v in self.model.state_dict().items()])

    def _decrypt_params(self, parameters: Parameters) -> NDArrays:
        params = zip(parameters.tensors,
                     [v.shape for k, v in self.model.state_dict().items()],
                     [v.dtype for k, v in self.model.state_dict().items()])

        return [torch.frombuffer(buffer=RsaCryptoAPI.decrypt_obj(self.__aes_key, param), dtype=dtype).reshape(shape).cpu().numpy() \
                for param, shape, dtype in params]

    def _encrypt_params(self, ndarrays: NDArrays) -> Parameters:
        enc_tensors = [RsaCryptoAPI.encrypt_numpy_array(self.__aes_key, arr) for arr in ndarrays]
        return Parameters(tensors=enc_tensors, tensor_type="")
    
    def _save_checkpoint(self, params):
        self.ckpt_name = f"ckpt_sym_{int(time.time())}.bin"
        with open(self.ckpt_name, 'wb') as f:
            pickle.dump(params, f)

    def _load_previous_checkpoint(self):
        with open(self.ckpt_name, 'rb') as f:
            params = pickle.load(f)
        return params

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
            ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if self.init_stage:
            # encrypt all params
            parameters = self._encrypt_params(parameters.tensors)
            self._save_checkpoint(parameters)
            self.init_stage = False

        if len(parameters.tensors) == 0:
            parameters = self._load_previous_checkpoint()

        fit_config = super().configure_fit(server_round, parameters, client_manager)

        for _, fit_ins in fit_config:
            public_key_pem, private_key_pem = RsaCryptoAPI.gen_rsa_key_pem()
            fit_ins.config['enc_key'] = RsaCryptoAPI.encrypt_aes_key(
                self.__aes_key, public_key_pem)
            fit_ins.config['private_key_pem'] = private_key_pem
            fit_ins.config['curr_round'] = server_round
            fit_ins.config['ds'] = self.dataset_name

        return fit_config

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
            ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.init_stage:
            parameters = self._encrypt_params(parameters.tensors)
            self.init_stage = False

        eval_config = super().configure_evaluate(server_round, parameters, client_manager)

        for _, ins in eval_config:
            public_key_pem, private_key_pem = RsaCryptoAPI.gen_rsa_key_pem()
            ins.config['enc_key'] = RsaCryptoAPI.encrypt_aes_key(
                self.__aes_key, public_key_pem)
            ins.config['private_key_pem'] = private_key_pem
            ins.config['ds'] = self.dataset_name
            ins.config['skip'] = (len(parameters.tensors) == 0)

        return eval_config

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # We deserialize using our custom method
        if self.init_stage:
            parameters_ndarrays = parameters.tensors
        else:
            parameters_ndarrays = self._decrypt_params(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        stragglers_mask = [res.metrics["is_straggler"] for _, res in results]
        stragglers_cnt = sum(stragglers_mask)
        if stragglers_cnt > 0:
            log(WARNING, f'Found {sum(stragglers_mask)} stragglers in this round; their weights will be discarded')

        # We deserialize each of the results with our custom method
        weights_results = [
            (self._decrypt_params(fit_res.parameters), fit_res.num_examples)
            for i, (_, fit_res) in enumerate(results) if not stragglers_mask[i]
        ]

        # We serialize the aggregated result using our custom method
        # TODO: Load encrypted model of previous checkpoint if all clients are stragglers
        parameters_aggregated = self._encrypt_params(
            aggregate(weights_results) #if stragglers_cnt < len(results) else self._load_previous_checkpoint()
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn and len(parameters_aggregated.tensors) > 0:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # TODO: Save new checkpoint here
        if len(parameters_aggregated.tensors) > 0:
            self._save_checkpoint(parameters_aggregated)

        return parameters_aggregated, metrics_aggregated