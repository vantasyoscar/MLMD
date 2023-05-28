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
        x = torch.Relu(self.layer1(x))
        x = torch.Sigmoid(self.layer2(x))
        x = self.layer3(x)
        return x


class DNN_sym(nn.Module):
    r'''
    Creating a DNN model that is permutation invariant. 
    Here we only implement permutation invariant DNN model for 2 different atoms.
    '''
    def __init__(self, atom,atom_list, embedding_dim: list[int], linear_layers: list[int]) -> None:
        super().__init__()
        self.atom = atom
        self.atom_list = atom_list
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed1 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)

        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed2 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)

        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed12 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)

        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]) ,*layers)
        self.output_layer = nn.Linear(linear_layers[-1], 3)

    def forward(self, x):
        if (self.atom == 1 and self.atom_list[0] == 1):
            g = self.embed1(x[0])
        elif (self.atom == 2 and self.atom_list[0] ==2):
            g = self.embed2(x[0])
        else:
            g = self.embed12(x[0])
        for i in range(1, len(x)):
            if (self.atom == 1 and self.atom_list[i] == 1):
                g_temp = self.embed1(x[i])
                g = torch.cat([g, g_temp], 0)
            elif (self.atom == 2 and self.atom_list[i] == 2):
                g_temp = self.embed2(x[i])
                g = torch.cat([g, g_temp], 0)
            else:
                g_temp = self.embed12(x[i])
                g = torch.cat([g, g_temp], 0)
        g = g.view(-1, 3)
        d = g.T @ x
        d = d.view(-1, 1).squeeze()
        d = self.linear_layers(d)
        out = self.output_layer(d)
        return out
    

if __name__ == "__main__":
    layers = [64, 128, 256, 3]
    model = DNN_sym(10, 8, layers)
    print(model)