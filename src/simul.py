from logging import WARNING, log
import os
import sys
from typing import List, Optional
import flwr as fl
import torch

# https://docs.python-guide.org/writing/structure/
# originally to be put outside src folder for other modules to call but ray fails
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from client import generate_client_fn
from dataset import prep_data_decentralized
from models.eval import get_evaluation_fn
from strategy.fhe_fed_avg import FheFedAvg
from strategy.sym_fed_avg import SymFedAvg

def run_simulation(
        data_path: str,
        mode: str = 'sym',
        batch_sz: int = 128,
        num_clients: int = 1,
        client_stragglers_prob: Optional[float | List[float]] = 0,
        client_proximal_mu: Optional[float | List[float]] = 0,
        serv_fraction_fit: float = 0.2,
        serv_fraction_eval: float = 0.2,
        serv_min_clients_avai: int = 1,
        serv_min_clients_fit: int = 1,
        serv_min_clients_eval: int = 1,
        serv_rounds: int = 3,
        gpu: bool = False,
        dashboard: bool = True
    ):

    dl_train, dl_val, dl_test = prep_data_decentralized(
        data_path=data_path, num_partitions=num_clients, batch_size=batch_sz)

    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not(torch.cuda.is_available()):
            log(WARNING, 'CUDA device is not available. Switching to CPU.')
    else:
        device = torch.device('cpu')

    fhe = (mode == 'fhe')
    if fhe:
        strategy = FheFedAvg(
            fraction_fit=serv_fraction_fit,
            fraction_evaluate=serv_fraction_eval,
            min_available_clients=serv_min_clients_avai,
            min_evaluate_clients=serv_min_clients_eval,
            min_fit_clients=serv_min_clients_fit,
            evaluate_fn=get_evaluation_fn(dl_test, device)
        )
    else:
        strategy = SymFedAvg(
            fraction_fit=serv_fraction_fit,
            fraction_evaluate=serv_fraction_eval,
            min_available_clients=serv_min_clients_avai,
            min_evaluate_clients=serv_min_clients_eval,
            min_fit_clients=serv_min_clients_fit,
            evaluate_fn=get_evaluation_fn(dl_test, device)
        )

    gen_client_fn = generate_client_fn(
        dl_trains=dl_train,
        dl_vals=dl_val,
        fhe=(mode == 'fhe'),
        device=device,
        straggler_prob=client_stragglers_prob,
        proximal_mu=client_proximal_mu
    )

    hist = fl.simulation.start_simulation(
        client_fn=gen_client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=serv_rounds),
        strategy=strategy,
        ray_init_args={
            "include_dashboard": dashboard
        }
    )

    return hist

if __name__ == '__main__':
    hist = run_simulation('data')
    print(hist)