# Flower + OpenFHE

## Getting started

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install OpenFHE

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

## Observability

Observability is currenly only available when running simulations with `python simul.py`.

Follow [the guide here](https://flower.ai/docs/framework/how-to-monitor-simulation.html). In summary:

- Install Prometheus and Grafana.
  - For Ubuntu Linux, `brew services start grafana` may not work. Install Grafana according to [this guide](https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/) and use `sudo systemctl start grafana-server` or `grafana-server --homepath HOME_PATH --config CONF_PATH`.
- Modify Grafana configuration to accept dashboards and datasources from Ray.
  - Allow anonymous view and set provisioning path to Ray folder.
- Start the services, they are accessible at:
  - Ray dashboard: [localhost:8265](http://localhost:8265)
  - Prometheus: [localhost:9090](http://localhost:9090)
  - Grafana: [localhost:3000](http://localhost:3000)

## Notes

- The transmission of the keys is done through model updates, not through modification of Flower's protocols.

- In the FHE scheme, the clients and the server will share the same crypto context, including its public and private keys. Hence, **the server will send to clients the crypto context and the 2 keys generated by OpenFHE directly**. These can be further encrypted by introducing the RSA scheme and adding RSA keys on top of FHE keys, or by conducting key exchange/derivation processes.

## How to customize it to your own model?

- Make sure your model is written in PyTorch and it can be dissected layer by layer (the crytographic services encrypt/decrypt the model by layers, i.e. `model.state_dict()`).

- Make necessary modifications to the code files under the `dataset` and `models` modules. Other modules should remain untouched (if you do not change the function signatures).

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
  --gpu                 Enable GPU training & inference (optional)
  --localhost           Run localhost only (optional)
  --host HOST           IP address or hostname of the server; cannot be used together with localhost
  --port PORT           Port number (default is 8080)

Data Configuration:
  --data_path DATA_PATH
                        Path to data folder
  --batch_size BATCH_SIZE
                        Data batch size (default is 32)
```