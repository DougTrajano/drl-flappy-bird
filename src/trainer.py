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

    def run(self, render: bool = False):
        """ Run the training process. """
        self.scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.print_range)  # last 100 scores

        for i in range(1, self.n_episodes+1):
            state = self._env.reset()
            score = 0
            done = False

            while not done:
                if render:
                    self._env.render()

                action = self._agent.act(state)

                next_state, reward, done, info = self._env.step(action)

                self._agent.step(state, action, reward, next_state, done, i)

                state = next_state.copy()

                score += reward

                if done:
                    break

            scores_window.append(score) # save most recent score
            self.scores.append(score) # save most recent score

            if self.verbose:
                print('\rEpisode {}\tAvg Score: {:.2f}'.format(
                    i, np.mean(scores_window)), end="")
                if i % self.print_range == 0:
                    self.logs.append({"episode": i, "score": np.mean(scores_window)})
                    print('\rEpisode {}\tAvg Score: {:.2f}'.format(
                        i, np.mean(scores_window)))

            if np.mean(scores_window) >= self.early_stop and i > 10:
                if self.verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAvg Score: {:.2f}'.format(
                        i, np.mean(scores_window)))
                break

        self._env.close()

        self.best_score = np.mean(scores_window)
        self.last_episode = i

        return True
