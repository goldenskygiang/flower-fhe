from collections import OrderedDict
from logging import WARNING, log
from typing import Callable
from models import test

import torch

def get_evaluation_fn(ds_name: str, dl_test, init_model_fn: Callable, device=None):
    '''
    Returns a function. The returned eval_fn() will be executed
    by the strategy at the end of each round to evaluate the
    stats of the global model
    '''
    def evaluate_fn(server_round: int, parameters, config):
        '''
        Executed by the strategy. Instantiates a model, replace its
        params with the global's. Then this model is evaluated on
        the test set (the whole central test set)
        '''
        if len(parameters) == 0:
            # skip evaluation since all clients were stragglers probably
            log(WARNING, "All clients are possibly stragglers in this round. Skip evaluation")
            return None, {'accuracy': None}

        model = init_model_fn()
        # set params
        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({
            k: torch.from_numpy(v.copy()) for k, v in params_dict
        })
        model.load_state_dict(state_dict, strict=True)

        # call test
        loss, accuracy = test(ds_name, model, dl_test, device)
        return loss, {'accuracy': accuracy}

    return evaluate_fn