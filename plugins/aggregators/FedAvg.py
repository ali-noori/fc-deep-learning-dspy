from utils.pytorch.optimizer import FedOptimizer
import numpy as np
from utils.utils import to_numpy


class CustomAggregator(FedOptimizer):
    """
    Supporting:
    * SMPC
    * CrossValidation
    * ReqAggData.WEIGHTS_N_SAMPLES
    """
    def __init__(self, **kwargs):
        super(CustomAggregator, self).__init__(**kwargs)
        '''
        params = [
        client 0:  [ (weights_fold0, n0_fold0),  (weights_fold1, n0_fold1) ],
        client 1:  [ (weights_fold0, n1_fold0),  (weights_fold1, n1_fold1) ],
        ]
        '''
        self.global_weights = []
        self.stopping_criteria = []

    def aggregate(self, params, **kwargs):
        # NEW; THIS CODE WAS ADDED AND INSTEAD OF THAT THERE WAS NOT CODE
        '''
        A. self.iteration += 1
        Count this meeting (1, 2, ..., up to max_iter).
        
        B. n_splits = len(params[0])
        How many folds? For you, 2.
        
        C.for each client:
        for each fold:
            sum_weights[fold] += client_weights[fold] * n_samples[fold]
            sum_n[fold]      += n_samples[fold]

        D. for each fold:
        updated_weights[fold] = sum_weights[fold] / sum_n[fold]

        E. self.global_weights = updated_weights
        '''
        self.iteration += 1
        n_splits = len(params[0])
        global_weights = [np.array(params[0][0][0], dtype='object') * 0] * n_splits
        total_n_samples = [0] * n_splits
        for client_models in params:
            for model_counter, (weights, n_samples) in enumerate(client_models):
                global_weights[model_counter] += np.array(weights, dtype='object') * n_samples
                total_n_samples[model_counter] += n_samples
        updated_weights = []
        for counter, (w, n) in enumerate(zip(global_weights, total_n_samples)):
            updated_weights.append(w / n)
        self.global_weights = updated_weights

    def aggregate_smpc(self, params):
        global_weights = [to_numpy(model) / total_n_sample for model, total_n_sample in params]
        self.weights = global_weights

    def post_aggregate(self, **kwargs):
        # OLD: metrics = kwargs['metrics']
        # OLD:iter_limit = self.iteration >= self.max_iter
        # OLD: self.stopping_criteria = [[iter_limit]] * len(metrics)

        # NEW
        '''
        The function is being called after the aggregation is done.
        It checks if the number of iterations has reached the maximum number of iterations.
        For each client that we have it send a sign that shows wether the client should stop or not.
        example:
        [[False, False], [True, True]]
        '''
        iter_limit = self.iteration >= self.max_iter
        n_models = len(self.global_weights)
        self.stopping_criteria = [iter_limit] * n_models
        # END NEW

    '''
    The FedAvg.py is module that it only returns the stopping criteria
    '''
    @property
    def stoppage(self):
        return self.stopping_criteria


    '''
    The FedAvg.py is module that it only returns the weights
    '''
    @property
    def weights(self):
        return self.global_weights

    '''
    The FedAvg.py is module that it only returns nothing for the config
    It can be used for other implementations of the aggregator.
    '''
    @property
    def config(self):
        return self.config

    '''
    The FedAvg.py is module that it only returns nothing for the gradients:
    It can be used for other implementations of the aggregator.
    '''
    @property
    def gradients(self):
        return None
