import torch
import torch.nn as nn
import neighbor_list

class MyDNN(nn.Module):
    def __init__(self, fwd):
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
    r'''
    Creating a DNN model that is permutation invariant.
    '''
    def __init__(self, num_embedding: int, embedding_dim: int, linear_layers: list[int]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        q = self.embedding(x)
        d = q @ x
        out = self.linear_layers(d)
        return out
    

if __name__ == "__main__":
    layers = [64, 128, 256, 3]
    model = DNN_sym(10, 8, layers)
    print(model)