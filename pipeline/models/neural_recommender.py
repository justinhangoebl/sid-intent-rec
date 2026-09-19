import torch.nn as nn

from pipeline.models.recommender import Recommender

class NeuralRecommender(Recommender, nn.Module):
    """Base class for all neural recommenders."""

    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, batch):
        """Compute the loss for a batch of samples."""
        raise NotImplementedError
    
    def collate(self, samples):
        """Collate a list of samples into a batch."""
        raise NotImplementedError
    
    def save(self, path):
        """Save the model to the specified path."""
        raise NotImplementedError
    
    def load(self, path):
        """Load the model from the specified path."""
        raise NotImplementedError