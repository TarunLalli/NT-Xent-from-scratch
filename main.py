import torch
import torch.nn as nn

class NTXent(nn.Module):
    def __init__(self):
        super().__init__()

    def lossfunction(self):
        ...