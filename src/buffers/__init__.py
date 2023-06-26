from .trajectory_buffer import TrajectoryReplayBuffer
from .prompt_buffer import PromptBuffer
from .continual_trajectory_buffer import ContinualTrajectoryReplayBuffer, CMDTrajectoryReplayBuffer
from .multi_domain_buffer import MultiDomainTrajectoryReplayBuffer


def make_buffer_class(kind): 
    if kind == "continual": 
        return ContinualTrajectoryReplayBuffer
    elif kind == "cmd": 
        return CMDTrajectoryReplayBuffer
    elif kind == "domain":
        return MultiDomainTrajectoryReplayBuffer
    elif kind == "prompt":
        return PromptBuffer
    return TrajectoryReplayBuffer
