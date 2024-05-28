import os
import argparse
import time

from utils.args_validator import fraction_validator, port_number_validator

def init_arguments():
    parser = argparse.ArgumentParser(
        description="Run Federated Learning server with model encryption mechanisms")

    parser.add_argument('--mode', choices=['fhe', 'sym'], required=True,
                        help='Choose either Fully Homomorphic Encryption (fhe) or Symmetric Encryption (sym) mode')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU for global evaluation (optional)')
    parser.add_argument('--localhost', action='store_true',
                        help='Run localhost only (optional)')
    parser.add_argument('--port', type=port_number_validator, default=8080,
                        help='Port number (default is 8080)')
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
    data_group.add_argument('--num_partitions', type=int, required=True,
                            help='Number of data partitions and clients')
    data_group.add_argument('--batch_size', type=int, default=32, 
                            help='Data batch size (default is 32)')
    data_group.add_argument('--cifar_ver', type=str, default='10',
                            help='CIFAR dataset version (optional, default is 10)')
    data_group.add_argument('--cifar_val_split', type=float, default=0.15,
                            help='CIFAR dataset validation split size (default is 0.15)')
    
    fl_group = parser.add_argument_group('Federated Learning Configuration')
    fl_group.add_argument('--server_rounds', type=int, default=5, 
                          help='Number of server rounds (default is 5)')
    fl_group.add_argument('--fraction_fit', type=fraction_validator, default=0.2, 
                          help='Fraction of fitting clients (default is 0.2, limit double value from 0 to 1)')
    fl_group.add_argument('--fraction_evaluate', type=fraction_validator, default=0.2, 
                          help='Fraction of evaluating clients (default is 0.2, limit double value from 0 to 1)')
    fl_group.add_argument('--min_available_clients', type=int, default=2, 
                          help='Minimum number of available clients to run federated learning (default is 2)')
    fl_group.add_argument('--min_evaluate_clients', type=int, default=2, 
                          help='Minimum number of evaluating clients (default is 2)')
    fl_group.add_argument('--min_fit_clients', type=int, default=2, 
                          help='Minimum number of fitting clients (default is 2)')
    
    fedprox_group = parser.add_argument_group('FedProx Configuration')
    fedprox_group.add_argument('--straggler_prob', type=fraction_validator, default=0,
                               help='Probability of being a Straggler node (default is 0, limit double value from 0 to 1)')
    fedprox_group.add_argument('--proximal_mu', type=float, default=0,
                               help='Proximal mu value (default is 0)')

    return parser.parse_args()

if __name__ == '__main__':
    args = init_arguments()

    processes = []

    sv_pid = os.fork()
    if (sv_pid == 0):
        # run server here
        from server import run_server
        run_server(args)
        exit(0)
    
    processes.append(sv_pid)

    time.sleep(10)

    clients = args.num_partitions
    for i in range(clients):
        pid = os.fork()

        if (pid == 0):
            # call client process
            from client import run_client
            serv_addr = f"127.0.0.1:{args.port}" if args.localhost else f"0.0.0.0:{args.port}"

            run_client(
                i, serv_addr, args.ds, args.data_path, args.num_partitions, args.batch_size, args.gpu,
                args.mode, args.server_rounds, args.straggler_prob, args.proximal_mu,
                args.cifar_ver, args.cifar_val_split, args.num_classes, args.threshold,
                args.model_choice, args.dropout, args.msg_max_sz, args.epochs)
            
            exit(0)

        processes.append(pid)

    for pid in processes:
        os.waitpid(pid, 0)

    print("*" * 20 + " SIMULATION COMPLETED " + "*" * 20)