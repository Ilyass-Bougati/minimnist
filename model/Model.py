import torch.nn as nn
import torch.nn.functional as f

class NN(nn.Module):
    def __init__(self, hidden_size):
        super(NN, self).__init__()
        self.l1 = nn.Linear(72, hidden_size)
        self.l2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        o1 = f.relu(self.l1(x))
        o2 = self.l2(o1)
        return o2