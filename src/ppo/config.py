from pydantic import BaseModel

class AgentConfig(BaseModel):
    state_size: int
    action_size: int
    seed: int = 1993
    memory_size: int = int(1e5)
    nb_hidden: tuple = (64, 64)
    gamma: float = 0.99
    lam: float = 0.97
    target_kl: float = 0.01
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_policy_iters: int = 10
    train_value_iters: int = 10
    clip_ratio: float = 0.2
    epsilon_enabled: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    model_dir: str = "./PPO.pt"