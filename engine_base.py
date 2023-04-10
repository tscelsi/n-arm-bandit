"""This file contains the base engine to run the n-arm bandit problem.
The Engine class will need to be extended to include a choosing algorithm.
The simplest would be a random action selection.
"""
import logging
import numpy as np
from bandit import Bandit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EngineError(Exception):
    pass


class Engine:
    def __init__(self, n_bandits=2000, n_actions=10, T: int=1000):
        """Initalise the engine

        Args:
            n_bandits (int): The number of bandit problems to create.
            n_actions (int): The number of actions the bandits can take.
            T (int): The number of time steps the engine should complete. Otherwise can be thought of
                as the number of times that we pull a bandit arm.
        """
        self.n_actions = n_actions
        self.n_bandits = n_bandits
        self.bandits = Bandit(self.n_bandits, self.n_actions)
        self.Q = self.bandits()  # initialise Q values (i.e. pull all arms once)
        self.N = np.ones((self.n_bandits, self.n_actions))
        self.T = T
        self.reward_history = []
        self.has_run = False
    
    def __str__(self):
        return self.__class__.__name__
    
    def choose(self) -> list[int]:
        """Returns a k length array of action indices.

        Raises:
            NotImplementedError: This class cannot be used itself. It needs to be inherited and this
            function needs to be implemented.
        """
        raise NotImplementedError('Need to implement choose method')

    def update(self, actions: list[int], rewards: list[float]):
        """A simple sample-average update algorithm to update the engines
        expectation of reward for a particular action.

        Args:
            reward (float): The reward value given by the bandit
            action (int): The action for which the reward value was seen
        """
        # just allows us to easily index all rows
        row_inds = range(self.n_bandits)
        self.N[row_inds, actions] += 1
        logger.debug(f"actions {actions} have been seen {self.N[:, actions]} times")
        old_expectations = self.Q[row_inds, actions]
        logger.debug(f"old reward expectations: {old_expectations}")
        new_expectations = old_expectations + 1/self.N[row_inds, actions] * (rewards - old_expectations)
        logger.debug(f"new reward expectations: {new_expectations}")
        self.Q[row_inds, actions] = new_expectations

    def run(self):
        t = 0
        while t < self.T:
            logger.debug(f"beginning step {t}")
            actions = self.choose(t)
            logger.debug(f"choosing actions {actions}")
            rewards = self.bandits(actions)
            self.reward_history.append(rewards)
            logger.debug(f"got reward {rewards} for action {actions}")
            self.update(actions, rewards)
            t += 1
        self.has_run = True

    def plot(self):
        if not self.has_run:
            raise EngineError('need to run n-arm bandit before plotting!')
        _, ax = plt.subplots()
        average_rewards = np.array(self.reward_history).mean(axis=1)
        ax.plot(range(self.T), average_rewards)
        plt.show()
