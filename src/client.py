import argparse
import os
import pandas as pd
import threading
import time
import psutil
from flwr.common.logger import log
from logging import INFO, WARNING
import numpy as np
from utils.args_validator import port_number_validator, host_validator, fraction_validator

def init_arguments():
    parser = argparse.ArgumentParser(
        description="Run Federated Learning client with model encryption mechanisms")

    parser.add_argument('--mode', choices=['fhe', 'sym'], required=True,
                        help='Choose either Fully Homomorphic Encryption (fhe) or Symmetric Encryption (sym) mode')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU training & inference (optional)')
    parser.add_argument('--localhost', action='store_true',
                        help='Run localhost only (optional)')
    parser.add_argument('--host', type=host_validator,
                        help='IP address or hostname of the server; cannot be used together with localhost')
    parser.add_argument('--port', type=port_number_validator, default=8080,
                        help='Port number (default is 8080)')
    parser.add_argument('--num_rounds', type=int, default=5,
                        help='Number of server rounds (default is 5)')
    
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_path', type=str, required=True,
                        help='Path to data folder')
    data_group.add_argument('--batch_size', type=int, default=32, 
                            help='Data batch size (default is 32)')
    
    fedprox_group = parser.add_argument_group('FedProx Configuration')
    fedprox_group.add_argument('--straggler_prob', type=fraction_validator, default=0,
                               help='Probability of being a Straggler node (default is 0, limit double value from 0 to 1)')
    fedprox_group.add_argument('--proximal_mu', type=float, default=0,
                               help='Proximal mu value (default is 0)')

    args = parser.parse_args()

    # Check for either host or localhost option
    if not args.localhost and (args.host is None):
        parser.error("Either --host or --localhost must be specified.")
    
    # Check for conflicting host options
    if args.host and args.localhost:
        parser.error("Conflicting options: --host and --localhost cannot be used together.")

    return args

def generate_client_fn(dl_trains, dl_vals, fhe: bool, device=None,
                       num_rounds = 5,
                       straggler_prob: float = 0,
                       proximal_mu: float = 0):
    from fl_clients.fhe_client import FheClient
    from fl_clients.sym_client import SymClient

    straggler_schedule = np.random.choice(
        [0, 1], size=(num_rounds), p=[1 - straggler_prob, straggler_prob])
    
    def client_fn(cid: str):
        cid = int(cid)
        train_dl = dl_trains[cid] if isinstance(dl_trains, list) else dl_trains
        eval_dl = dl_vals[cid] if isinstance(dl_vals, list) else dl_vals

        if fhe:
            return FheClient(
                cid=cid,
                dl_train=train_dl,
                dl_val=eval_dl,
                device=device,
                straggler_sched=straggler_schedule,
                proximal_mu=proximal_mu
            )
        else:
            return SymClient(
                cid=cid,
                dl_train=train_dl,
                dl_val=eval_dl,
                device=device,
                straggler_sched=straggler_schedule,
                proximal_mu=proximal_mu
            )

    return client_fn

stop_event = threading.Event()

def measure_current_process_stats(metrics_data: list, localhost: bool):
    process = psutil.Process(os.getpid())

    old_net_info = psutil.net_io_counters(pernic=localhost, nowrap=False)
    if localhost:
        old_net_info = old_net_info['lo']
    
    while not stop_event.is_set():
        # Memory usage
        mem_info = process.memory_info()
        memory_usage = mem_info.rss
        
        # Network usage
        net_info = psutil.net_io_counters(pernic=localhost, nowrap=False)
        if localhost:
            net_info = net_info['lo']

        net_usage_sent = max(0, net_info.bytes_sent - old_net_info.bytes_sent)
        net_usage_recv = max(0, net_info.bytes_recv - old_net_info.bytes_recv)

        metrics_data.append({
            "Timestamp": time.time(),
            "MemoryUsage": memory_usage,
            "NetSent": net_usage_sent,
            "NetRecv": net_usage_recv
        })

        log(INFO, f"Mem={memory_usage}, NetSent={net_usage_sent}, NetRecv={net_usage_recv}")
        old_net_info = net_info

        time.sleep(1)

def run_client(
        server_addr: str,
        data_path: str,
        batch_size: int,
        gpu: bool,
        mode: str,
        num_rounds: int,
        straggler_prob: float = 0,
        proximal_mu: float = 0):
    
    import torch
    import flwr as fl
    from dataset import prep_data
    from torch.utils.data import DataLoader

    dl_train, dl_val, dl_test = prep_data(data_path=data_path)

    dl_train = DataLoader(dl_train, batch_size=batch_size)
    dl_val = DataLoader(dl_val, batch_size=batch_size)
    
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not(torch.cuda.is_available()):
            log(WARNING, 'CUDA device is not available. Switching to CPU.')
    else:
        device = torch.device('cpu')

    log(INFO, f'Using device: {device}')
    
    client_fn = generate_client_fn(
        dl_train, dl_val, mode == 'fhe', device, num_rounds, straggler_prob, proximal_mu)

    log(INFO, f"Client connecting to server at {server_addr}")

    if max(straggler_prob, proximal_mu) > 0:
        log(INFO, f"Client using FedProx with straggler_prob={straggler_prob}, mu={proximal_mu}")

    fl.client.app.start_client(
        server_address=server_addr,
        client_fn=client_fn
    )

if __name__ == '__main__':
    args = init_arguments()

    server_addr = f"127.0.0.1:{args.port}" if args.localhost else f"{args.host}:{args.port}"

    metrics_data = []

    metrics_thread = threading.Thread(target=measure_current_process_stats, args=(metrics_data, args.localhost))
    metrics_thread.start()

    run_client(
        server_addr, args.data_path, args.batch_size, args.gpu,
        args.mode, args.num_rounds, args.straggler_prob, args.proximal_mu)
    
    stop_event.set()
    metrics_thread.join()

    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f"{time.time()}_{args.mode}_SVR-{args.num_rounds}_SP-{args.straggler_prob}_PM-{args.proximal_mu}.csv"
    metrics_df.to_csv(metrics_filename, index=False)