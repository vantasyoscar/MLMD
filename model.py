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

    Args::
    atom: int: A index to distinguish the core atom. (e.g. core atom H -> 1)
    atom_list: list[int]: A list of the index of the neighbor atoms. (e.g. [H, O, O, H, H] -> [1, 2, 2, 1, 1])
    embedding_dim: list[int]: Determine the structure of the full-connect embedding net structure. (e.g. [20, 40] -> [nn.Linear(3, 20), 
     nn.Linear(20, 40) ]).
    linear_layers: list[int]: Determine the structure of the full-connect fitting net structure. Same as embedding_dim.

    Input::
    input.size() = (num_neighbor_atoms, 3). Each row is the cartesian position of the atom.

    Output::
    output.size() = (3). Gives out the velocity(displacement) of the core atom.

    Example::
    Please see the example in "__main__".

    '''
    def __init__(self, atom: int, atom_list: list[int], embedding_dim: list[int], linear_layers: list[int]) -> None:
        super().__init__()
        self.atom = atom
        self.atom_list = atom_list
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed1 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed2 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
        self.embed12 = nn.Sequential(nn.Linear(3, embedding_dim[0]), *layers)
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]) ,*layers)
        self.output_layer = nn.Linear(linear_layers[-1], 3)

    def new_atom_list(self, atom, atom_list):
        self.atom = atom
        self.atom_list = atom_list

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
        g = g.view(len(x), -1)
        d = g.T @ x
        d = d.view(-1, 1).squeeze()
        d = self.linear_layers(d)
        out = self.output_layer(d)
        return out
    

if __name__ == "__main__":
    embed_layers = [20, 40]
    linear_layers = [64, 128, 256]
    atom = 1
    atom_list = [1, 2, 1, 2, 1, 2, 1, 1, 2, 1]
    model = DNN_sym(atom, atom_list, embed_layers, linear_layers)
    print(model)
    model.to('cpu')
    input = torch.rand(len(atom_list), 3)
    input.to('cpu')
    target = torch.rand(3)
    target.to('cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    output = model(input)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # An illustration of permutation invariant
    new_input = torch.zeros_like(input)
    permu_index = torch.LongTensor([2, 1, 0, 3, 4, 5, 6, 7, 8, 9])      ### Change position 0 and 2 atom_list will not change.
    new_input[permu_index] = input
    diff_input = torch.zeros_like(input)
    permu_index = torch.LongTensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])      ### Change position 0 and 1 atom_list will change.
    diff_input[permu_index] = input
    print(input[0:3, :])
    print(new_input[0:3, :])
    print(diff_input[0:3, :])
    print(model(input), model(new_input), model(diff_input))       ### Here we verify that the two input will not change.