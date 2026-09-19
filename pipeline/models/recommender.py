class Recommender:
    """Base class for all recommenders."""

    def __init__(self, config):
        self.config = config
    
    def generate(self, users, history, k, filter_seen=True):
        """Generate top-k recommendations for a batch of users."""
        raise NotImplementedError
    
    def get_config(self):
        """Return the model configuration."""
        return self.config