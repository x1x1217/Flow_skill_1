import torch

from models.flow_prior import FlowTeacher, FlowStudent, flow_loss, distill_loss

class Flow_BC(object):
    """
    Offline train condition:
        cond = [s0, noisy_a0]
        
    Online sample condition:
        cond = [o, n]
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        hidden_dim=256,
        time_dim=16,
        flow_steps=,
    ):
        