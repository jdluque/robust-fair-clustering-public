from enum import Enum, auto


class FairType(Enum):
    """Which fairness type to use."""
    PROB = auto()  # Probabilistic
    ROBUST = auto()  # Robust clustering
    VANILLA = auto() # No fairness
    DET = auto()  # Deterministic clustering
