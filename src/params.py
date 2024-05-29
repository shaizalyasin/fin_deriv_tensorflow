from abc import ABC


class Params(ABC):
    pass


class TreeParams(Params):
    """Parameters specific to Binomial Tree method."""
    def __init__(self, num_steps: int):
        self.num_steps = num_steps


class MCParams(Params):
    """Parameters specific to Monte Carlo method."""
    def __init__(self, num_paths: int, time_steps: int):
        self.num_paths = num_paths
        self.time_steps = time_steps
