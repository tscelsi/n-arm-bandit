import numpy as np
from engine_base import Engine


class EngineSimple(Engine):
    def choose(self, *args, **kwargs) -> np.ndarray:
        """Chooses an action randomly for each bandit."""
        return np.random.randint(0, self.n_actions - 1, size=self.n_bandits)


class EngineArgmax(Engine):
    def choose(self, *args, **kwargs) -> np.ndarray:
        """Chooses the action with the highest expected return for each bandit.
        """
        return np.argmax(self.Q, axis=1)


class EngineEpsilonGreedy(Engine):
    def __init__(self, *args, **kwargs):
        epsilon = kwargs.pop('epsilon', 0.1)
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)
    
    def __str__(self):
        return self.__class__.__name__ + f"({self.epsilon})"

    def choose(self, *args, **kwargs):
        """Chooses the action with the highest expected return most of the time (exploiting)
        but on some rare occasions chooses a random action (exploring).
        """
        is_greedy = np.random.uniform(size=self.n_bandits)
        should_exploit = is_greedy > self.epsilon
        # initialise bandit actions as random action selection
        choices = np.random.randint(0, self.n_actions - 1, size=self.n_bandits)
        greedy_choices = np.argmax(self.Q, axis=1)
        # replace random choices with greedy choices <epsilon*100> % of the time
        choices[should_exploit] = greedy_choices[should_exploit]
        return choices


class EngineUCB(Engine):
    def __init__(self, *args, **kwargs):
        c = kwargs.pop('c', 2)
        self.c = c
        super().__init__(*args, **kwargs)
    
    def choose(self, t):
        """Chooses the action based on an Upper Confidence Bound strategy (UCB).
        """
        return np.argmax(self.Q + self.c * np.sqrt(np.log(t + 1) / self.N), axis=1)
