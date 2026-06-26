import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.enabled = True
    cudnn.benchmark = False if deterministic else True
    cudnn.deterministic = True if deterministic else False
