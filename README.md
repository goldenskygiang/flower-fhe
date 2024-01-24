# Flower + OpenFHE

## Getting started

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install OpenFHE dependencies

```sh
sudo apt-get install -y clang libomp5 libomp-dev cmake libgmp3-dev libntl-dev libomp-dev autoconf

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

cd ./openfhe-development
mkdir build 
cd build
cmake .. -DMATHBACKEND=6 -DWITH_OPENMP=ON -DWITH_NATIVEOPT=ON -DBUILD_SHARE=ON
make -j 8
sudo make install
```

### Install OpenFHE-python binding

```sh
cd ./openfhe-python
mkdir build
cd build
cmake ..
make -j 8
make install
```

## Server

`python server.py [options]`

### Server options

```sh
$ python server.py --help

usage: server.py [-h] --mode {fhe,sym} [--https] [--gpu] [--localhost] [--port PORT] --data_path DATA_PATH [--num_partitions NUM_PARTITIONS] [--batch_size BATCH_SIZE]
                 [--server_rounds SERVER_ROUNDS] [--fraction_fit FRACTION_FIT] [--fraction_evaluate FRACTION_EVALUATE] [--min_available_clients MIN_AVAILABLE_CLIENTS]
                 [--min_evaluate_clients MIN_EVALUATE_CLIENTS] [--min_fit_clients MIN_FIT_CLIENTS]

Run Federated Learning server with model encryption mechanisms

options:
  -h, --help            show this help message and exit
  --mode {fhe,sym}      Choose either Fully Homomorphic Encryption (fhe) or Symmetric Encryption (sym) mode
  --https               Enable HTTPS (optional)
  --gpu                 Enable GPU for global evaluation (optional)
  --localhost           Run localhost only (optional)
  --port PORT           Port number (default is 8080)

Data Configuration:
  --data_path DATA_PATH
                        Path to data folder
  --num_partitions NUM_PARTITIONS
                        Number of data partitions (default is 100)
  --batch_size BATCH_SIZE
                        Data batch size (default is 32)

Federated Learning Configuration:
  --server_rounds SERVER_ROUNDS
                        Number of server rounds (default is 5)
  --fraction_fit FRACTION_FIT
                        Fraction of fitting clients (default is 0.2, limit double value from 0 to 1)
  --fraction_evaluate FRACTION_EVALUATE
                        Fraction of evaluating clients (default is 0.2, limit double value from 0 to 1)
  --min_available_clients MIN_AVAILABLE_CLIENTS
                        Minimum number of available clients to run federated learning (default is 2)
  --min_evaluate_clients MIN_EVALUATE_CLIENTS
                        Minimum number of evaluating clients (default is 2)
  --min_fit_clients MIN_FIT_CLIENTS
                        Minimum number of fitting clients (default is 2)
```

## Client

`python client.py [options]`

### Client options

```sh
$ python client.py --help

usage: client.py [-h] --mode {fhe,sym} [--https] [--gpu] [--localhost] [--host HOST] [--port PORT] --data_path DATA_PATH
                 [--num_partitions NUM_PARTITIONS] [--batch_size BATCH_SIZE]

Run Federated Learning client with model encryption mechanisms

options:
  -h, --help            show this help message and exit
  --mode {fhe,sym}      Choose either Fully Homomorphic Encryption (fhe) or Symmetric Encryption (sym) mode
  --https               Enable HTTPS (optional)
  --gpu                 Enable GPU training & inference (optional)
  --localhost           Run localhost only (optional)
  --host HOST           IP address or hostname of the server; cannot be used together with localhost
  --port PORT           Port number (default is 8080)

Data Configuration:
  --data_path DATA_PATH
                        Path to data folder
  --num_partitions NUM_PARTITIONS
                        Number of data partitions (default is 100)
  --batch_size BATCH_SIZE
                        Data batch size (default is 32)
```