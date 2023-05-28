import torch
import torch.nn as nn

class MyDNN(nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.layer1 = nn.Linear(62*3*fwd, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 62*3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x


class DNN_sym(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        