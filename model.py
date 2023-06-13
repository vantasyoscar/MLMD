import torch
import torch.nn as nn
import neighbor_list
import numpy as np

class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = 0
    
    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total= self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1


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
        self.activation = nn.LeakyReLU()
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
            layers.append(self.activation)
        self.embed1 = nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation, *layers)
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
            layers.append(self.activation)
        self.embed2 = nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation, *layers)
        layers = []
        for i in range(len(embedding_dim)-1):
            layers.append(nn.Linear(embedding_dim[i], embedding_dim[i+1]))
            layers.append(self.activation)
        self.embed12 = nn.Sequential(nn.Linear(3, embedding_dim[0]),self.activation, *layers)
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            layers.append(self.activation)
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]),self.activation ,*layers)
        self.output_layer = nn.Linear(linear_layers[-1], 3)
        #self.output_acti = nn.Sigmoid()

    def new_atom_list(self, atom, atom_list):
        self.atom = atom
        self.atom_list = atom_list

    def forward(self, x):
        if (self.atom == 1 and self.atom_list[0] == 1):
            g = self.embed1(x[0])
        elif (self.atom == 2 and self.atom_list[0] == 2):
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
        #out = self.output_acti(out)
        return out
    

class DNN_sym_new(nn.Module):
    r'''
    Creating a DNN model that is permutation invariant. 
    The new version of the DDN_sym model now can be used to fit any atom type number.

    Args::
    atom_num: An integer that indicate how many different atom type are in the system.
    atom: int: A index to distinguish the core atom, the first index should be 0. (e.g. core atom H -> 0)
    atom_list: list[int]: A list of the index of the neighbor atoms. (e.g. [H, O, O, H, H] -> [0, 1, 1, 0, 0])
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
    def __init__(self, atom_num: int, atom: int, atom_list: list[int], embedding_dim: list[int], linear_layers: list[int]) -> None:
        super().__init__()
        self.atom = atom
        self.atom_list = atom_list
        self.embed_dict = nn.ModuleDict()
        self.activation = nn.LeakyReLU()
        layers = []
        for i in range(atom_num):
            for j in range(i, atom_num):
                for k in range(len(embedding_dim)-1):
                    layers.append(nn.Linear(embedding_dim[k], embedding_dim[k+1]))
                    layers.append(self.activation)
                self.embed_dict.add_module(name=f'embed{str(i)}{str(j)}', 
                                           module=nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation, *layers))
                layers = []
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            layers.append(self.activation)
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]), self.activation ,*layers)
        self.output_layer = nn.Linear(linear_layers[-1], 3)

    def new_atom_list(self, atom, atom_list):
        self.atom = atom
        self.atom_list = atom_list

    def forward(self, x):
        if (self.atom >= self.atom_list[0]):
            g = self.embed_dict[f'embed{str(self.atom_list[0])}{str(self.atom)}'](x[0])
        else:
            g = self.embed_dict[f'embed{str(self.atom)}{str(self.atom_list[0])}'](x[0])
        for i in range(1, len(x)):
            if (self.atom >= self.atom_list[i]):
                g_temp = self.embed_dict[f'embed{str(self.atom_list[i])}{str(self.atom)}'](x[i])
                g = torch.cat([g, g_temp], dim=0)
            else:
                g_temp = self.embed_dict[f'embed{str(self.atom)}{self.atom_list[i]}'](x[i])
                g = torch.cat([g, g_temp], dim=0)
        g = g.view(len(x), -1)
        d = g.T @ x
        d = d.view(-1, 1).squeeze()
        d = self.linear_layers(d)
        out = self.output_layer(d)
        return out
    

class DNN_full(nn.Module):
    def __init__(self, atom_list: list[int], cut_off: float, embedding_dim: list[int], linear_layers: list[int], lattice: list[float] = None) -> None:
        super().__init__()
        self.atom_num = len(atom_list)
        self.atom_type_num = len(np.unique(atom_list))
        self.cut_off = cut_off
        self.lattice = lattice
        self.embed_dict = nn.ModuleDict()
        self.linear_list = nn.ModuleList()
        self.activation = nn.LeakyReLU()
        layers = []
        for i in range(self.atom_type_num):
            for j in range(i, self.atom_type_num):
                for k in range(len(embedding_dim)-1):
                    layers.append(nn.Linear(embedding_dim[k], embedding_dim[k+1]))
                    layers.append(self.activation)
                self.embed_dict.add_module(name=f'embed{str(i)}{str(j)}', 
                                           module=nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation, *layers))
                layers = []
        layers = []
        for j in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[j], linear_layers[j+1]))
            layers.append(self.activation)
        for i in range(self.atom_num):
            self.linear_list.append(nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]), self.activation, *layers, nn.Linear(linear_layers[-1], 3)))
        
    def forward(self, x):
        # neighboring list generation:
        x_ext = x - x[0]
        for i in range(1, self.atom_num):
            pass


if __name__ == "__main__":
    embed_layers = [20, 40]
    linear_layers = [64, 128, 256]
    atom = 0
    atom_list = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    model = DNN_sym_new(2, atom, atom_list, embed_layers, linear_layers)
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