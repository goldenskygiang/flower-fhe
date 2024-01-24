import argparse
import re

def host_validator(value):
    ip_pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
    hostname_pattern = re.compile(r'^[a-zA-Z0-9.-]+$')

    if ip_pattern.match(value) or hostname_pattern.match(value):
        return value
    raise argparse.ArgumentTypeError(f"{value} is not a valid IP address or hostname.")

def fraction_validator(value):
    value = float(value)
    if 0 <= value <= 1:
        return value
    raise argparse.ArgumentTypeError(
        f"{value} is not a valid fraction. It must be a double value between 0 and 1.")

def port_number_validator(value):
    value = int(value)
    if 0 <= value <= 65535:
        return value
    raise argparse.ArgumentError(
        f"{value} is not a valid port number. It must be an integer value between 0 and 65535.")