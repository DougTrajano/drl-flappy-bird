import time
from gym import Env
from src.base import Agent


def play_env(agent: Agent, env: Env, fps: int = 30, render: bool = False):
    """
    Play an environment with an agent.

    Args:
    - agent: Agent to play the environment.
    - env: Environment to play.
    - fps: Frames per second.
    - render: Render the environment.
    """
    score = 0
    
    state = env.reset()
    while True:
        if render:
            env.render()

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        score += reward
        state = next_state

        time.sleep(1 / fps)
        
        if done:
            break
        
    print(f"Score: {score}")
    env.close()