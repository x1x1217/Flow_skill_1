import numpy as np
import torch
import swanlab


def _to_scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 1:
            return value.item()
        return value.cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SwanLabWriter:
    """Small SummaryWriter-compatible wrapper around SwanLab."""

    def __init__(
        self,
        project,
        experiment_name,
        workspace=None,
        config=None,
        logdir=None,
        mode=None,
        tags=None,
        group=None,
    ):
        self.run = swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=experiment_name,
            config=config,
            logdir=logdir,
            mode=mode,
            tags=tags,
            group=group,
        )

    def add_scalar(self, tag, scalar_value, global_step=None):
        data = {tag: _to_scalar(scalar_value)}
        if global_step is None:
            swanlab.log(data)
        else:
            swanlab.log(data, step=int(global_step))

    def close(self):
        swanlab.finish()
