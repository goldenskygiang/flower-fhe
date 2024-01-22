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

- `--https` (optional)

- `--mode=[fhe/sym]`

## Client

`python client.py [options]`

### Client options

- `--server=[IP_ADDRESS]`

- `--gpu` (optional)

- `--mode=[fhe/sym]`
