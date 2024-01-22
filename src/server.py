import argparse
import flwr as fl

from strategy.fhe_fed_avg import FheFedAvg
from strategy.sym_fed_avg import SymFedAvg

from model.eval import get_evaluation_fn

def init_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["fhe", "sym"], default="fhe")

    return parser.parse_args()

if __name__ == 'main':
    args = init_arguments()

    if args.mode == 'fhe':
        strategy = FheFedAvg(
            fraction_fit=0.2,  # sample 10% of clients each round for local training
            fraction_evaluate=0.5, # after each round, sample 20% of clients to assess performance
            min_available_clients=2, # total number of clients available
            evaluate_fn=get_evaluation_fn(dl_test) # callback to a fn that the strategy can use
        )
    else:
        strategy = SymFedAvg()

    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )