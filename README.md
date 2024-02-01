# Flower + OpenFHE

## Getting started

### Linux

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Install OpenFHE

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

#### Install OpenFHE-python binding

```sh
cd ./openfhe-python
mkdir build
cd build
cmake ..
make -j 8
make install
```

### Windows

TOTALLY INCOMPLETE GUIDE, WOULD NOT WORK RIGHT NOW.

#### Install OpenFHE

- Download and install [MSYS2](https://www.msys2.org/).

- Open **MSYS2 MinGW64** and run the following:

```sh
pacman -Syu
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-cmake
pacman -S make
pacman -S git
pacman -S autoconf
```

- Download [NTL Unix](https://libntl.org/doc/tour-unix.html) and unzip.

- Build NTL Unix in MSYS2 terminal.

```sh
cd src
./configure
make
make install
```

- Clone this repo (including submodules with `--recurse-submodules`).

- `cd openfhe-development` and run the following, also in MSYS2 shell **as administrator**:

```sh
mkdir build
cd build
cmake .. -DMATHBACKEND=6 -DWITH_OPENMP=ON -DWITH_NATIVEOPT=ON -DBUILD_SHARE=ON -DNTL_INCLUDE_DIR=/usr/local/include -DNTL_LIBRARIES=/usr/local/lib
make -j 8
make install
```

Usually OpenFHE will be installed at ` C:/Program Files (x86)/OpenFHE`.

#### Install OpenFHE-python binding

- Add these lines in `openfhe-python/CMakeLists.txt`.

```
include_directories( ${NTL_INCLUDE_DIR} )
link_libraries( ${NTL_LIBDIR} )
```

- Run the following in MSYS2 MinGW shell.

  - If the `cmake` throws error about missing `pybind11`, find `pybind11` CMake files and add option `-Dpybind11_DIR` (usually at `venv/Lib/site-packages/pybind11/share/cmake/pybind11`)

```sh
export PATH=$PATH:'/c/Program Files (x86)/OpenFHE/lib:/usr/local/lib'

python -m venv venv
source venv/Scripts/activate
pip install -r requirements-win.txt

python -m pip install ninja

cd openfhe-python
mkdir build
cd build
cmake .. -DOpenFHE_DIR='/c/Program Files (x86)/OpenFHE' -DNTL_INCLUDE_DIR=/usr/local/include -DNTL_LIBDIR=/usr/local/lib/libntl.a
ninja
ninja install
```

### How to know if you setup correctly?

```sh
$ source venv/bin/activate
$ python
>>> from openfhe import *
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
  --num_partitions NUM_PARTITIONS
                        Number of data partitions (default is 100)
  --batch_size BATCH_SIZE
                        Data batch size (default is 32)
```