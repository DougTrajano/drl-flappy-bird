import time
import logging
from gym import Env
from src.base import Agent

_logger = logging.getLogger(__name__)

def play_env(agent: Agent, env: Env, fps: int = 30, render: bool = False):
    """
    Play an environment with an agent.

    Args:
    - agent: Agent to play the environment.
    - env: Environment to play.
    - fps: Frames per second.
    - render: Render the environment.
    """
    _logger.info("Playing environment...")

    score = 0
    
    state = env.reset()
    while True:
        if render:
            env.render()
            time.sleep(1 / fps)
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state.copy()
        score += reward
        if done:
            break
        print(f"\rScore: {score:.2f}", end="")
        
    print(f"\rScore: {score:.2f}")
    env.close()
    
    _logger.info("Environment finished.")

    return score