# Random RL Agent
import random
import logging
import numpy as np

_logger = logging.getLogger(__name__)

class Agent:
    def __init__(self,  state_size: int, action_size: int, seed: int = 1993):
        """Initialize a Random Agent object.

        Parameters
        - state_size: dimension of each state
        - action_size: dimension of each action
        - seed: random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        _logger.info("Random Agent initialized")

    def act(self, state: np.ndarray):
        """
        Returns a random action

        Args:
        - state: current state

        Returns
        - action: action to take
        """
        return random.choice(np.arange(self.action_size)).astype(int)

    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, i: int):
        """
        Performs a step from the environment.
        In the Random Agent, this function does nothing.

        Args:
        - state: current state
        - action: action taken
        - reward: reward received
        - next_state: next state
        - done: if the episode is over
        - i: current step
        """
        pass

    def prep_state(state: np.ndarray) -> np.ndarray:
        """
        Preprocess state before feeding it to the act function.
        You can replace this function with your own implementation.
        
        Args:
        - state: state to be preprocessed
        
        Returns:
        - state: preprocessed state
        """
        return state