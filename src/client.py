import argparse
import sys
from utils.args_validator import port_number_validator, host_validator

def init_arguments():
    parser = argparse.ArgumentParser(
        description="Run Federated Learning client with model encryption mechanisms")

    parser.add_argument('--mode', choices=['fhe', 'sym'], required=True,
                        help='Choose either Fully Homomorphic Encryption (fhe) or Symmetric Encryption (sym) mode')
    # parser.add_argument('--https', action='store_true',
    #                     help='Enable HTTPS (optional)')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU training & inference (optional)')
    parser.add_argument('--localhost', action='store_true',
                        help='Run localhost only (optional)')
    parser.add_argument('--host', type=host_validator,
                        help='IP address or hostname of the server; cannot be used together with localhost')
    parser.add_argument('--port', type=port_number_validator, default=8080,
                        help='Port number (default is 8080)')
    
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, required=True,
                        help='Path to data folder')
    data_group.add_argument('--batch_size', type=int, default=32, 
                            help='Data batch size (default is 32)')
    
    args = parser.parse_args()

    # Check for either host or localhost option
    if not args.localhost and (args.host is None):
        parser.error("Either --host or --localhost must be specified.")
    
    # Check for conflicting host options
    if args.host and args.localhost:
        parser.error("Conflicting options: --host and --localhost cannot be used together.")

    return args

def generate_client_fn(dl_trains, dl_vals, fhe: bool, device=None):    
    from fl_clients.fhe_client import FheClient
    from fl_clients.sym_client import SymClient
    
    def client_fn(cid: str):
        if fhe:
            return FheClient(
                cid=cid,
                dl_train=dl_trains,
                dl_val=dl_vals,
                device=device)
        else:
            return SymClient(
                cid=cid,
                dl_train=dl_train,
                dl_val=dl_vals,
                device=device
            )

    return client_fn

if __name__ == '__main__':
    args = init_arguments()

    import torch
    import flwr as fl
    from flwr.common.logger import log
    from logging import INFO, WARNING
    from dataset import prep_data
    from torch.utils.data import DataLoader

    dl_train, dl_val, dl_test = prep_data(data_path=args.data_path)

    dl_train = DataLoader(dl_train, batch_size=args.batch_size)
    dl_val = DataLoader(dl_val, batch_size=args.batch_size)
    
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not(torch.cuda.is_available()):
            log(WARNING, 'CUDA device is not available. Switching to CPU.')
    else:
        device = torch.device('cpu')

    log(INFO, f'Using device: {device}')
    
    client_fn = generate_client_fn(dl_train, dl_val, args.mode == 'fhe', device)

    server_addr = f"127.0.0.1:{args.port}" if args.localhost else f"{args.host}:{args.port}"

    log(INFO, f"Client connecting to server at {server_addr}")

    hist = fl.client.app.start_client(
        server_address=server_addr,
        client_fn=client_fn
    )

    print(f"{hist}")