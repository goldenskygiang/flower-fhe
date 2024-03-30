# Outside wrapper script to execute simulations

import os
import sys

# https://docs.python-guide.org/writing/structure/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from client import generate_client_fn

# TODO: Handle end-to-end simulation calls here, including running time and network usage

print('Loaded!')