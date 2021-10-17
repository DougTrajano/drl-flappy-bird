import time
import logging
import numpy as np
from gym import Env
from collections import deque
from src.base import Agent

_logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, agent: Agent, env: Env, n_episodes: int = 8000,
                 print_range: int = 100, early_stop: int = 300, verbose: bool = False):
        """
        Reinforcement Learning trainer.

        Args:
        - agent: Agent to train.
        - env: Environment to train on.
        - n_episodes: maximum number of training episodes
        - print_range: range to print partials results
        - eps_start: starting value of epsilon, for epsilon-greedy action selection
        - eps_end: minimum value of epsilon
        - eps_decay: multiplicative factor (per episode) for decreasing epsilon
        - early_stop: Stop training when achieve a defined score respecting n_episodes.
        - verbose: Print partial results.
        """
        self._agent = agent
        self._env = env
        self.n_episodes = n_episodes
        self.print_range = print_range
        self.early_stop = early_stop
        self.verbose = verbose

        self.scores = []
        self.logs = []
        self.best_score = None
        self.last_episode = None

        _logger.info('Trainer initialized.')

    def run(self, max_timestep: int = None, render: bool = False,
            fps: int = 30, logs_callback: callable = None):
        """
        Run the training session.
        
        Args:
        - max_timestep: maximum number of timesteps per episode.
        - render: Render the environment.
        - fps: Frames per second (only if render is True).
        - logs_callback: Callback function that provides logs (returns a str or None).
        """
        _logger.info('Starting training...')

        self.scores = []
        scores_window = deque(maxlen=self.print_range)

        for ep in range(1, self.n_episodes+1):
            _logger.info(f"Episode {ep}/{self.n_episodes}")

            state = self._env.reset()
            score = 0
            done = False

            if max_timestep is not None:
                for _ in range(max_timestep):
                    if render:
                        self._env.render()
                        time.sleep(1 / fps)
                    action = self._agent.act(state)
                    next_state, reward, done, _ = self._env.step(action)
                    self._agent.step(state, action, reward, next_state, done, ep)
                    state = next_state.copy()
                    score += reward
                    if done:
                        break                    
            else:
                while not done:
                    if render:
                        self._env.render()
                        time.sleep(1 / fps)
                    action = self._agent.act(state)
                    next_state, reward, done, _ = self._env.step(action)
                    self._agent.step(state, action, reward, next_state, done, ep)
                    state = next_state.copy()
                    score += reward
                    if done:
                        break

            scores_window.append(score)
            self.scores.append(score)

            if self.verbose:
                log_msg = f"\rEpisode {ep}\tAvg Score: {np.mean(scores_window):.2f}"
                if logs_callback and logs_callback():
                    log_msg += "\t" + logs_callback()
                print(log_msg, end="")
                if ep % self.print_range == 0:
                    self.logs.append({"episode": ep, "avg_score": np.mean(scores_window)})
                    print(log_msg)

            if np.mean(scores_window) >= self.early_stop and ep > 10:
                if self.verbose:
                    print(f"\nEnvironment solved in {ep:d} episodes!\tAvg Score: {np.mean(scores_window):.2f}")
                break

        self._env.reset()

        self.best_score = np.mean(scores_window)
        self.last_episode = ep

        _logger.info(f"Training finished.\nBest score: {self.best_score:.2f}")

        return True
