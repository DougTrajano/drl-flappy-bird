import torch
import random
import numpy as np
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedMemory:
    """
    Fixed-size memory to store experience tuples with sampling weights.
    PRIORITIZED EXPERIENCE REPLAY - https://arxiv.org/pdf/1511.05952.pdf
    """
    def __init__(self, memory_size: int, batch_size: int, alpha: float = 0.7):
        """Initialize a ReplayMemory object.

        Parameters
        - memory_size: maximum size of memory
        - batch_size: size of each training batch
        - alpha: determines how much prioritization is used
        """
        self.capacity = memory_size
        self.memory = deque(maxlen=memory_size)
        self.alpha = alpha
        self.batch_size = batch_size
        self.priority = deque(maxlen=memory_size)
        self.probabilities = np.zeros(memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new experience to memory.

        Args:
        - state: current state
        - action: action taken
        - reward: reward received
        - next_state: next state
        - done: whether the episode is done
        """

        priority_max = max(self.priority) if self.memory else 1
        e = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(e)
        self.priority.append(priority_max)
    
    def sample(self, beta: float = 0.4):
        """Sample a batch of experiences from prioritized memory.
        
        Args:
        - beta: determines how much prioritization is used
        """
        self.update_probabilities()
        index = np.random.choice(range(self.capacity), self.batch_size, replace=False, p=self.probabilities)
        experiences = [self.memory[i] for i in index]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        # Calculate sampling weights
        sampling_weights = (self.__len__()*self.probabilities[index])**(-beta)
        sampling_weights = sampling_weights / np.max(sampling_weights)
        sampling_weights = torch.from_numpy(sampling_weights).float().to(device)
        
        return (states, actions, rewards, next_states, dones, index, sampling_weights)
    
    def update_probabilities(self):
        """Update sampling probabilities."""
        probabilities = np.array([i**self.alpha for i in self.priority])
        self.probabilities[range(len(self.priority))] = probabilities
        self.probabilities /= np.sum(self.probabilities)
        
    def update_priority(self, indexes: np.ndarray, losses: np.ndarray):
        """
        Update priorities of sampled experiences.

        Args:
        - indexes: indexes of sampled experiences
        - losses: losses of sampled experiences
        """
        for index, loss in zip(indexes, losses):
            self.priority[index] = loss.data
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class ReplayMemory:
    """Fixed-size memory to store experience tuples."""

    def __init__(self, memory_size: int, batch_size: int):
        """Initialize a ReplayMemory object.

        Parameters:
        - memory_size: maximum size of memory
        - batch_size: size of each training batch
        """
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)