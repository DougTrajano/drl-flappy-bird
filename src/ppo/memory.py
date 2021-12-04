import logging
import numpy as np
import scipy.signal
import tensorflow as tf

_logger = logging.getLogger(__name__)


class Memory(object):
    """
    Memory to store experience that uses Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, state_size: int, action_size: int, memory_size: int,
                 gamma: float = 0.99, lam: float = 0.95):
        """Initialize Memory object.

        Parameters:
        - state_size: size of the state vector.
        - action_size: size of the action vector.
        - memory_size: size of the memory.
        - gamma: discount factor.
        - lam: lambda parameter for GAE-Lambda.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.lam = lam

        self.states = np.zeros((memory_size, state_size), dtype=np.float32)
        self.actions = np.zeros(memory_size, dtype=np.int32)
        self.advantages = np.zeros(memory_size, dtype=np.float32)
        self.rewards = np.zeros(memory_size, dtype=np.float32)
        self.returns = np.zeros(memory_size, dtype=np.float32)
        self.values = np.zeros(memory_size, dtype=np.float32)
        self.logprobabilities = np.zeros(memory_size, dtype=np.float32)

        self.pointer = 0
        self.trajectory_start_index = 0

        _logger.info(f"Memory initialized with capacity: {self.memory_size} items.")

    def add(self, state: np.ndarray, action: tf.Tensor, reward: int, value: tf.Tensor, logprobability: tf.Tensor):
        """
        Add a new experience to memory.

        Args:
        - state: state vector
        - action: action vector
        - reward: reward
        - value: value of the next state
        - logprobability: log probability of the action
        """
        _logger.debug("Adding new experience to memory.")

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.values[self.pointer] = value
        self.logprobabilities[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value: int = 0):
        """
        Finish the trajectory by computing advantage estimates and rewards-to-go

        Args:
        - last_value: value of the last state
        """
        _logger.debug(f"Finishing trajectory with last value: {last_value}.")

        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages[path_slice] = self.discounted_cum_sums(
            deltas, self.gamma * self.lam
        )
        self.returns[path_slice] = self.discounted_cum_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

        _logger.debug("Finished trajectory,")

    def discounted_cum_sums(self, rewards: np.ndarray, discount: float):
        """
        Compute the discounted cumulative sums of a list of rewards.

        Args:
        - rewards: vector of rewards.
        - discount: discount factor.

        Returns:
        - Discounted cumulative sums of rewards.
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], rewards[::-1], axis=0)[::-1]

    def get(self):
        """
        Get all data of the buffer and normalize the advantages.
        """
        _logger.debug("Getting data from memory.")
        
        self.pointer = 0
        self.trajectory_start_index = 0

        advantage_mean, advantage_std = (
            np.mean(self.advantages),
            np.std(self.advantages),
        )

        self.advantages = (
            self.advantages - advantage_mean) / advantage_std

        return self.states, self.actions, self.advantages, self.returns, self.logprobabilities
