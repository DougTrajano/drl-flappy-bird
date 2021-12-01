import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Any
from keras.models import load_model

from src.base import Agent
from src.ppo.memory import Memory


class Agent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int = 1993,
                 memory_size: int = int(1e5), nb_hidden: tuple = (64, 64),
                 gamma: float = 0.99, lam: float = 0.97,
                 target_kl: float = 0.01,
                 policy_lr: float = 3e-4, value_lr: float = 1e-3,
                 train_policy_iters: int = 80, train_value_iters: int = 80,
                 clip_ratio: float = 0.2,
                 epsilon_enabled: bool = True, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 **kwargs):
        """
        Initialize a Proximal Policy Optimization (PPO) agent.

        Parameters:
        - state_size: dimension of each state.
        - action_size: dimension of each action.
        - seed: random seed.
        - memory_size: size of the replay memory.
        - nb_hidden: number of hidden layers in the network.
        - gamma: discount factor. (Always between 0 and 1.).
        - lam: lambda for GAE-Lambda.
        - target_kl: KL divergence between target and current policy. This will get used for early stopping in learn function. (Usually small, 0.01 or 0.05.)
        - policy_lr: learning rate for the policy optimizer.
        - value_function_lr: learning rate for the value function optimizer.
        - train_policy_iters: Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)
        - train_value_iters: Number of gradient descent steps to take on value function per epoch.
        - clip_ratio: clipping ratio for the policy objective.
        - epsilon_enabled: if True, use epsilon-greedy action selection.
        - epsilon_start: initial value for the epsilon parameter.
        - epsilon_end: final value for the epsilon parameter.
        - epsilon_decay: decay rate for the epsilon parameter.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.memory_size = memory_size
        self.gamma = gamma
        self.lam = lam
        self.target_kl = target_kl
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.train_policy_iters = train_policy_iters
        self.train_value_iters = train_value_iters
        self.clip_ratio = clip_ratio
        self.epsilon_enabled = epsilon_enabled
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize epsilon
        if self.epsilon_enabled:
            self.epsilon = self.epsilon_start
        else:
            self.epsilon = 0.0

        self.memory = Memory(self.state_size, self.action_size,
                             self.memory_size, self.gamma, self.lam)

        # Initialize the actor and the critic as keras models
        state_input = keras.Input(shape=(self.state_size,), dtype=tf.float32)
        logits = self.mlp(state_input, list(nb_hidden) +
                          [self.action_size], tf.tanh, None)

        self.actor = keras.Model(inputs=state_input, outputs=logits)
        self.value = tf.squeeze(
            self.mlp(state_input, list(nb_hidden) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=state_input, outputs=self.value)

        # Initialize the policy and the value function optimizers
        self.policy_opt = Adam(learning_rate=self.policy_lr)
        self.value_opt = Adam(learning_rate=self.value_lr)

    def logs(self):
        """
        Logs the agent's performance.
        You can replace this function with your own implementation.

        Returns:
        - A string with the log message.
        """
        return f"Epsilon: {self.epsilon:.2f}"

    def act(self, state: np.ndarray):
        """
        Returns a random action.

        Args:
        - state: current state.

        Returns
        - action: action to take.
        """
        # Preprocessing state
        state = self.prep_state(state)

        # Sample action from actor
        logits = self.actor(state)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        # Get the value and log-probability of the action
        self.temp_value = self.critic(state)
        self.temp_logprob = self.compute_logprobabilities(logits, action)

        # Epsilon-greedy action selection
        if self.epsilon_enabled and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = action[0].numpy()

        if self.epsilon_enabled:
            self.decay_eps()

        return action

    def step(self, state: np.ndarray, action: int, reward: int, done: bool, **kwargs):
        """
        Performs a step from the environment.
        In the Random Agent, this function does nothing.

        Args:
        - state: current state.
        - action: action taken.
        - reward: reward received.
        - done: if the episode is over.
        """
        # Preprocessing state
        state = self.prep_state(state)

        # Convert action to tensor
        action = tf.convert_to_tensor([action], dtype=tf.int64)

        self.memory.add(state, action, reward,
                        self.temp_value, self.temp_logprob)

        self.temp_value = None
        self.temp_logprob = None

        if done:
            last_value = 0 if done else self.critic(state)
            self.memory.finish_trajectory(last_value)

            experiences = self.memory.get()
            self.learn(experiences)

    def learn(self, experiences: Tuple[Any]):
        """
        Update value parameters using given batch of experience tuples.

        Args:
        - experiences: tuple of (states, actions, advantages, returns, logprobs) tuples.
        """
        states, actions, advantages, returns, logprobs = experiences

        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.train_policy_iters):
            kl = self.train_policy(states, actions, logprobs, advantages)
            if kl > 1.5 * self.target_kl:
                break  # Early Stopping

        # Update the value function
        for _ in range(self.train_value_iters):
            self.train_value_function(states, returns)

    def decay_eps(self):
        """
        Decay epsilon-greedy used for action selection.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def save_model(self, path: str = "/PPO"):
        """
        Save the model to the given path.

        Args:
        - path: path to save the model
        """
        # remove / at the end of the path
        if path[-1] == "/":
            path = path[:-1]

        self.actor.save(f"{path}/actor")
        self.critic.save(f"{path}/critic")

    def load_model(self, path: str = "/PPO"):
        """
        Load the model from the given path.

        Args:
        - path: path to load the model
        """
        # remove / at the end of the path
        if path[-1] == "/":
            path = path[:-1]

        self.actor = load_model(f"{path}/actor")
        self.critic = load_model(f"{path}/critic")

    def mlp(self, x: int, nb_hidden: List[int],
            activation: callable = tf.tanh, output_activation: callable = None):
        """
        Build a feedforward neural network.

        Args:
        - x: input for the network
        - nb_hidden: list with the number of units in each hidden layer
        - activation: activation function
        - output_activation: output activation function

        Returns:
        - output tensor
        """
        for size in nb_hidden[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=nb_hidden[-1], activation=output_activation)(x)

    def compute_logprobabilities(self, logits: tf.Tensor, actions: tf.Tensor):
        """
        Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor).

        Args:
        - logits: the output of the actor
        - actions: the actions to take

        Returns:
        - log-probabilities of the actions
        """
        logprobs_all = tf.nn.log_softmax(logits)
        
        logprob = tf.reduce_sum(
            tf.one_hot(actions, self.action_size) * logprobs_all, axis=1
        )

        return logprob

    def train_policy(self, states: np.ndarray, actions: np.ndarray,
                     logprobabilities: np.ndarray, advantages: np.ndarray):
        """
        Train the policy by maxizing the PPO-Clip objective.

        Args:
        - states: batch of states.
        - actions: batch of actions.
        - logprobabilities: batch of log-probabilities.
        - advantages: batch of advantages.
        """
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.compute_logprobabilities(self.actor(states), actions)
                - logprobabilities
            )
            min_advantage = tf.where(
                advantages > 0,
                (1 + self.clip_ratio) * advantages,
                (1 - self.clip_ratio) * advantages,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )
        policy_grads = tape.gradient(
            policy_loss, self.actor.trainable_variables)
        self.policy_opt.apply_gradients(
            zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobabilities
            - self.compute_logprobabilities(self.actor(states), actions)
        )
        kl = tf.reduce_sum(kl)

        return kl

    def train_value_function(self, states, returns):
        """
        Train the value function by regression on mean-squared error.

        Args:
        - states: batch of states.
        - returns: batch of returns.
        """
        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean((returns - self.critic(states)) ** 2)
        value_grads = tape.gradient(
            value_loss, self.critic.trainable_variables)
        self.value_opt.apply_gradients(
            zip(value_grads, self.critic.trainable_variables))
