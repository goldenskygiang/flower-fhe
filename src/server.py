import argparse
from models import generate_model_fn
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

    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--ds', choices=['pascal', 'cifar'], required=True,
                            help='Dataset name (pascal or cifar)')
    data_group.add_argument('--data_path', type=str, required=True,
                            help='Path to data folder')
    data_group.add_argument('--num_partitions', type=int, default=100, 
                            help='Number of data partitions (default is 100)')
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

    return parser.parse_args()

def run_server(args):
    import torch
    import flwr as fl
    from flwr.common.logger import log
    from logging import INFO, WARNING

    from models.eval import get_evaluation_fn
    from dataset import prep_data_decentralized
    from strategy.fhe_fed_avg import FheFedAvg
    from strategy.sym_fed_avg import SymFedAvg

    dl_train, dl_val, dl_test = prep_data_decentralized(
        ds_name=args.ds, data_path=args.data_path,
        num_partitions=args.num_partitions, batch_size=args.batch_size,
        cifar_ver=args.cifar_ver, cifar_val_split=args.cifar_val_split)
    
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not(torch.cuda.is_available()):
            log(WARNING, 'CUDA device is not available. Switching to CPU.')
    else:
        device = torch.device('cpu')

    log(INFO, f'Using device: {device}')

    init_model_fn = generate_model_fn(
        ds=args.ds,
        num_classes=args.num_classes,
        threshold=args.threshold,
        model_choice=args.model_choice,
        dropout=args.dropout,
    )

    if args.mode == 'fhe':
        strategy = FheFedAvg(
            init_model_fn=init_model_fn,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_available_clients=args.min_available_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_fit_clients=args.min_fit_clients,
            evaluate_fn=get_evaluation_fn(args.ds, dl_test, init_model_fn, device),
            dataset_name=args.ds
        )
    elif args.mode == 'sym':
        strategy = SymFedAvg(
            init_model_fn=init_model_fn,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_available_clients=args.min_available_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_fit_clients=args.min_fit_clients,
            evaluate_fn=get_evaluation_fn(args.ds, dl_test, init_model_fn, device),
            dataset_name=args.ds
        )

    server_addr = f"127.0.0.1:{args.port}" if args.localhost else f"0.0.0.0:{args.port}"

    log(INFO, f"Server listening on {server_addr}")

    hist = fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=args.server_rounds),
        strategy=strategy,
        grpc_max_message_length=args.msg_max_sz
    )

    print(f"{hist}")

if __name__ == '__main__':
    args = init_arguments()
    run_server(args)