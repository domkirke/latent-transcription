# -*- coding: utf-8 -*-
import torch
from numpy.random import permutation
import numpy as np

def length(array):
    if issubclass(type(array), list):
        return len(array)
    elif issubclass(type(array), np.ndarray):
        return array.shape[0]

class DataLoader(object):
    def __init__(self, dataset, batch_size, tasks=None, partition=None, ids=None, batch_cache_size = 1, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size

        self.batch_cache_size = batch_cache_size 
        if partition is None:
            random_indices = permutation(length(dataset.data)) 
        else:
            partition_ids = dataset.partitions[partition]
            random_indices = partition_ids[permutation(len(partition_ids))]
        if not ids is None:
            random_indices = ids
            
        # make random batches
        if self.batch_size:
            n_batches = len(random_indices)//batch_size
            random_ids = np.split(random_indices[:n_batches*batch_size], len(random_indices)//batch_size)
        else:
            random_ids = [random_indices]
        
        # make caches of batches
        self.random_ids = [random_ids[i:i+batch_cache_size] for i in range(0, len(random_ids), batch_cache_size)]
        self.tasks = tasks
    
    
    def __iter__(self):
        for batch_id in range(len(self.random_ids)):
            # load current cache
            if issubclass(type(self.dataset.data), list):
                current_data = [[self.dataset.data[i][d] for d in range(len(self.dataset.data[i]))] for i in self.random_ids[batch_id]]
            else:
                current_data = [self.dataset.data[i] for i in self.random_ids[batch_id]]
                
            for b, ids in enumerate(self.random_ids[batch_id]):
                self.current_ids = ids
                x = current_data[b]
                if not self.tasks is None:
                    y = {t:self.dataset.metadata[t][self.current_ids] for t in self.tasks}
                    yield x, y 
                else:
                    yield x, None
                    
#                yield self.transform(self.dataset.data[self.random_ids[i]]), None
        


class MixtureLoader(DataLoader):
    def __init__(self, datasets, batch_size, tasks=None, partition=None, ids=None, batch_cache_size = 1, random_mode='uniform', *args, **kwargs):
        self.batch_size = batch_size
        self.tasks = None
        self.partition = None
        self.batch_cache_size = batch_cache_size
        self.loaders = []
        self.random_mode = random_mode
        for d in datasets:
            self.loaders.append(DataLoader(d, batch_size, tasks, partition, ids, batch_cache_size, *args, **kwargs))

    def get_random_weights(self, n_batches, n_weights, mode):
        if mode == 'uniform':
            weights = np.random.rand(n_batches, n_weights)
        elif mode == 'normal':
            weights = np.random.randn(n_batches, n_weights)
        elif mode == 'constant':
            weights = np.ones((n_batches, n_weights))
        elif mode == 'bernoulli':
            weights = torch.distributions.Bernoulli(torch.full((n_batches, n_weights),0.5)).sample().numpy()
        return weights
    
    def __iter__(self, *args, **kwargs):
        # load iterators
        iterators = [loader.__iter__() for loader in self.loaders]
        finished = False
        random_mode = kwargs.get('random_mode', self.random_mode)
        try:
            # launch the loop
            while not finished:
                x = []; y = []; self.current_ids = []
                # iterate through loaders
                for i, iterator in enumerate(iterators):
                    x_tmp, y_tmp = next(iterator)
                    x.append(x_tmp); y.append(y_tmp); self.current_ids.append(self.loaders[i].current_ids)

                min_size = min([x_tmp.shape[0] for x_tmp in x])
                x = [x_tmp[:min_size] for x_tmp in x]
                self.current_ids = [cid[:min_size] for cid in self.current_ids]
                # make a mixture of data
                self.random_weights = self.get_random_weights(x[0].shape[0], len(x), random_mode)
                final_mixture = np.zeros_like(x[0])
                for i in range(len(x)):
                    final_mixture += (np.expand_dims(self.random_weights[:, i], 1) * x[i])
                yield final_mixture, x, y
        except StopIteration:
            # stop iteration
            finished = True
