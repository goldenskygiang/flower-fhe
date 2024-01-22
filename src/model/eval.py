from model import get_model

def get_evaluation_fn(dl_test):
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
        model = get_model()
        # set params
        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({
            k: torch.from_numpy(v.copy()) for k, v in params_dict
        })
        model.load_state_dict(state_dict, strict=True)

        # call test
        loss, accuracy = test(model, dl_test)
        return loss, {'accuracy': accuracy}

    return evaluate_fn