import argparse
import os
from typing import Callable
import pandas as pd
import threading
import time
import psutil
from flwr.common.logger import log
from logging import INFO, WARNING
import numpy as np
from models import generate_model_fn
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
    parser.add_argument('--msg_max_sz', type=int, default=2 * 1000 * 1000 * 1000,
                    help='Maximum gRPC message size in bytes (default is 2147483648 ~ 2GB)')

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--num_classes', type=int, default=20,
                             help='Number of output classes. 20 for PascalVOC multilabel, [10, 100] for Cifar multiclass')
    model_group.add_argument('--threshold', type=float, default=0.5,
                             help='Prediction threshold for Binary Classification (or multi-label)')
    model_group.add_argument('--model_choice', choices=['mobilenet', 'resnet', 'mnasnet'], default='mobilenet',
                             help="The backbone CNN model. Either 'mobilenet' or 'resnet' atm")
    model_group.add_argument('--dropout', type=float, default=0.4,
                             help="Dropout probability for the classification head's dropout layer")
    model_group.add_argument('--epochs', type=int, default=1,
                             help='Number of training epochs per round (default is 1)')
    
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--ds', choices=['pascal', 'cifar'], required=True,
                            help='Dataset name (pascal or cifar)')
    data_group.add_argument('--data_path', type=str, required=True,
                            help='Path to data folder')
    data_group.add_argument('--num_partitions', type=int, default=100, 
                            help='Number of data partitions (default is 100)')
    data_group.add_argument('--batch_size', type=int, default=32, 
                            help='Data batch size (default is 32)')
    data_group.add_argument('--cid', type=int, required=True,
                            help='Client ID (must be a non-negative integer in the range [0, num_partitions) )')
    data_group.add_argument('--cifar_ver', type=str, default='10',
                            help='CIFAR dataset version (optional, default is 10)')
    data_group.add_argument('--cifar_val_split', type=float, default=0.15,
                            help='CIFAR dataset validation split size (default is 0.15)')
    
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

    if args.cid < 0 or args.cid >= args.num_partitions:
        parser.error("Invalid client ID, must be an integer in the range [0, num_partitions)")

    return args

def generate_client_fn(client_id: int, 
                       dl_train, dl_val, fhe: bool, init_model_fn: Callable,
                       device=None,
                       num_rounds = 5,
                       straggler_prob: float = 0,
                       proximal_mu: float = 0,
                       epochs: int=1):
    from fl_clients.fhe_client import FheClient
    from fl_clients.sym_client import SymClient

    straggler_schedule = np.random.choice(
        [0, 1], size=(num_rounds), p=[1 - straggler_prob, straggler_prob]).tolist()
    
    log(INFO, f"Straggler schedule: {straggler_schedule}")
    
    def client_fn(cid: str):
        if fhe:
            log(INFO, "FHE client creating")
            return FheClient(
                cid=client_id,
                dl_train=dl_train,
                dl_val=dl_val,
                device=device,
                straggler_sched=straggler_schedule,
                proximal_mu=proximal_mu,
                init_model_fn=init_model_fn,
                epochs=epochs
            )
        else:
            return SymClient(
                cid=client_id,
                dl_train=dl_train,
                dl_val=dl_val,
                device=device,
                straggler_sched=straggler_schedule,
                proximal_mu=proximal_mu,
                init_model_fn=init_model_fn,
                epochs=epochs
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

        # log(INFO, f"Mem={memory_usage}, NetSent={net_usage_sent}, NetRecv={net_usage_recv}")
        old_net_info = net_info

        time.sleep(1)

def run_client(
        cid: int,
        server_addr: str,
        ds_name: str,
        data_path: str,
        num_partitions: int,
        batch_size: int,
        gpu: bool,
        mode: str,
        num_rounds: int,
        straggler_prob: float = 0,
        proximal_mu: float = 0,
        cifar_ver: str='10',
        cifar_val_split: float=0.15,
        num_classes: int=20,
        threshold: float=0.5,
        model_choice: str='mobilenet',
        dropout: float=0.4,
        msg_max_sz: int=2000000000,
        epochs: int=1):
    
    import torch
    import flwr as fl
    from dataset import prep_data_decentralized

    dl_train, dl_val, _ = prep_data_decentralized(
        ds_name=ds_name, data_path=data_path,
        num_partitions=num_partitions, batch_size=batch_size,
        cifar_ver=cifar_ver, cifar_val_split=cifar_val_split)
    
    dl_train = dl_train[cid]
    dl_val = dl_val[cid]
    
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not(torch.cuda.is_available()):
            log(WARNING, 'CUDA device is not available. Switching to CPU.')
    else:
        device = torch.device('cpu')

    log(INFO, f'Using device: {device}')

    init_model_fn = generate_model_fn(
        ds=ds_name,
        num_classes=num_classes,
        threshold=threshold,
        model_choice=model_choice,
        dropout=dropout,
    )
    
    client_fn = generate_client_fn(
        cid, dl_train, dl_val, mode == 'fhe', init_model_fn,
        device, num_rounds, straggler_prob, proximal_mu, epochs)

    log(INFO, f"Client connecting to server at {server_addr}")

    if max(straggler_prob, proximal_mu) > 0:
        log(INFO, f"Client using FedProx with straggler_prob={straggler_prob}, mu={proximal_mu}")

    metrics_data = []

    metrics_thread = threading.Thread(
        target=measure_current_process_stats,args=(metrics_data, server_addr.startswith("127.0.0.1")))
    metrics_thread.start()

    fl.client.app.start_client(
        server_address=server_addr,
        client_fn=client_fn,
        grpc_max_message_length=msg_max_sz
    )

    stop_event.set()
    metrics_thread.join()

    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = f"exp_{int(time.time())}_{mode}_C{cid}_SVR-{num_rounds}_SP-{straggler_prob}_PM-{proximal_mu}.csv"
    metrics_df.to_csv(metrics_filename, index=False)

if __name__ == '__main__':
    args = init_arguments()

    server_addr = f"127.0.0.1:{args.port}" if args.localhost else f"{args.host}:{args.port}"

    run_client(
        args.cid, server_addr, args.ds, args.data_path, args.num_partitions, args.batch_size, args.gpu,
        args.mode, args.num_rounds, args.straggler_prob, args.proximal_mu,
        args.cifar_ver, args.cifar_val_split, args.num_classes, args.threshold,
        args.model_choice, args.dropout, args.msg_max_sz, args.epochs)