"""Here is the code for a bandit that for each of its actions,
draws its rewards from a Normal distribution with variance 1 and mean
set so that one action is more desirable than another."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Bandit:
    def __init__(self, k=2000, n=10):
        """Initialises k n-arm bandit problems by computing their mean reward
        sampling from a N(0, 1) distribution.

        Args:
            k (int, optional): The number of bandit problems to create. Defaults to 2000.
            n (int, optional): The number of actions the bandits can take. Defaults to 10.
        """
        self.n_actions = n
        self.n_bandits = k
        self.means = np.random.normal(0, 1, (k, n))  # initialise the bandit arm means

    def __call__(self, actions: np.array=None):
        if actions is None:
            # should only be called on initialisation for the first set
            return np.random.normal(self.means)
        means = self.means[range(self.n_bandits), actions]
        logger.debug(f"means: {means}")        
        return np.random.normal(means, 1)
