# RL - Term Project

This repository contains our term project in the Reinforcement Learning course in [PUCRS](https://pucrs.br).

## Environment

The environment used in this project is an implementation of the Flappy Bird game using the OpenAI Gym library.

We are using the version `FlappyBird-v0` yields simple numerical information about the game's state as observations. The yielded attributes are the:

- horizontal distance to the next pipe;
- difference between the player's y position and the next hole's y position.

<p align="center">
  <img align="center" 
       src="https://github.com/Talendar/flappy-bird-gym/blob/main/imgs/yellow_bird_playing.gif?raw=true" 
       width="200"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" 
       src="https://github.com/Talendar/flappy-bird-gym/blob/main/imgs/red_bird_start_screen.gif?raw=true" 
       width="200"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img align="center" 
       src="https://github.com/Talendar/flappy-bird-gym/blob/main/imgs/blue_bird_playing.gif?raw=true" 
       width="200"/>
</p>

See more in [Talendar/flappy-bird-gym: An OpenAI Gym environment for the Flappy Bird game](https://github.com/Talendar/flappy-bird-gym)

## RL Algorithms

- [Deep Q-Network (DQN)](notebooks/DQN.ipynb)
- [Proximal Policy Optimization (PPO)](notebooks/PPO.ipynb)

## Changelog

See the [GitHub Tags](https://github.com/DougTrajano/pucrs-rl-term-project/tags) for a history of notable changes to this project.

## License

This software is licensed under the Apache 2.0 [LICENSE](LICENSE).