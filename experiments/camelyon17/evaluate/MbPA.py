import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import copy

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class ReplayMemory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size):
        self.memory = {}
        self.size = size
        
    def get_size(self):
        return len(self.memory) 
    
    def push(self, keys, logits):
       
        dimension = 1024*3*3
        avg = []
       
        for i, key in enumerate(keys):
           
            if len(self.memory.keys())>self.size:
                self.memory.pop(list(self.memory)[0]) 
            self.memory.update(
                {key.reshape(dimension).tobytes(): (logits[i])})
                
    def _prepare_batch(self, sample):
        
        
        ensemble_prediction = sample[0]
       
        for logit in sample:
            ensemble_prediction = ensemble_prediction + logit
       
        ensemble_prediction = ensemble_prediction - sample[0]
        ensemble_prediction = ensemble_prediction / len(sample)
        return torch.FloatTensor(ensemble_prediction)
    
    
    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []
       
        dimension = 1024*3*3
        keys = keys.reshape(len(keys), dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, dimension)
        
        for key in keys:
           
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]
            batch = self._prepare_batch(neighbours)
    
            samples.append(batch)
        
        return torch.stack(samples)
