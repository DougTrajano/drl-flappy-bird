from pydantic import BaseModel

class AgentConfig(BaseModel):
    state_size: int
    action_size: int
    seed: int = 1993
    nb_hidden: tuple = (64, 64)
    learning_rate: float = 0.0005
    memory_size: int = 100000
    prioritized_memory: bool = True
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.001
    small_eps: float = 0.03
    update_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99995
    model_dir: str = "./DuelingDQN.pt"
