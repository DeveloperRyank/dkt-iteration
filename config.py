from dataclasses import dataclass


@dataclass
class Config:
    num_skills: int = 12          # set by the dataset
    hidden_size: int = 64
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 20
    seed: int = 42
    device: str = "cpu"
