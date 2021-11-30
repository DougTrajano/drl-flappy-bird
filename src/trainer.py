import os
import time
import logging
import numpy as np
from gym import Env
from collections import deque
from src.base import Agent

_logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, agent: Agent, env: Env, n_episodes: int = 60000,
                 print_range: int = 1000, early_stop: int = 120,
                 max_timestep: int = None, fps: int = 30,
                 render: bool = False, verbose: bool = False, **kwargs):
        """
        Reinforcement Learning trainer.

        Args:
        - agent: Agent to train.
        - env: Environment to train on.
        - n_episodes: maximum number of training episodes.
        - print_range: range to print partials results.
        - early_stop: Stop training when achieve a defined score respecting n_episodes.
        - max_timestep: maximum number of timesteps per episode.
        - fps: frames per second (for rendering).
        - render: render the environment.
        - verbose: Print partial results.
        """
        self._agent = agent
        self._env = env
        self.n_episodes = n_episodes
        self.print_range = print_range
        self.early_stop = early_stop
        self.max_timestep = max_timestep
        self.fps = fps
        self.render = render
        self.verbose = verbose

        self.scores = []
        self.logs = []
        self.best_score = None
        self.best_episode = None
        self.last_score = None
        self.last_episode = None

        _logger.info('Trainer initialized.')

    def run(self, logs_callback: callable = None, save_best_model: bool = False,
            output_path: str = "models/best_model.pt", **kwargs):
        """
        Run the training session.
        
        Args:
        - logs_callback: Callback function that provides logs (returns a str or None).
        - save_best_model: Save the best model.
        - output_path: Path to save the model.
        """
        _logger.info('Starting training...')

        self.scores = []
        scores_window = deque(maxlen=self.print_range)

        for ep in range(1, self.n_episodes+1):
            _logger.info(f"Episode {ep}/{self.n_episodes}")

            state = self._env.reset()
            score = 0
            done = False

            if self.max_timestep is not None:
                for _ in range(self.max_timestep):
                    if self.render:
                        self._env.render()
                        time.sleep(1 / self.fps)
                    action = self._agent.act(state)
                    next_state, reward, done, _ = self._env.step(action)
                    self._agent.step(state=state, action=action, reward=reward,
                                     next_state=next_state, done=done, episode=ep)
                    state = next_state.copy()
                    score += reward
                    if done:
                        break                    
            else:
                while not done:
                    if self.render:
                        self._env.render()
                        time.sleep(1 / self.fps)
                    action = self._agent.act(state)
                    next_state, reward, done, _ = self._env.step(action)
                    self._agent.step(state=state, action=action, reward=reward,
                                     next_state=next_state, done=done, episode=ep)
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

                if self.best_score is None or self.best_score < np.mean(scores_window):
                    self.best_score = np.mean(scores_window)
                    self.best_episode = ep
                    if save_best_model:
                        if output_path is not None:
                            output_folder = os.path.dirname(output_path)
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            self._agent.save_model(output_path)

            if np.mean(scores_window) >= self.early_stop and ep > self.print_range:
                if self.verbose:
                    print(f"\nEnvironment solved in {ep:d} episodes!\tAvg Score: {np.mean(scores_window):.2f}")
                break

        self._env.reset()

        self.last_score = np.mean(scores_window)
        self.last_episode = ep

        _logger.info(f"Training finished.\nBest score: {self.best_score:.2f}\tLast score: {self.last_score:.2f}")

        return True
