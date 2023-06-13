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
            total = self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1


class DnnSym(nn.Module):
    r"""
    Creating a DNN model that is permutation invariant.
    Here we only implement permutation invariant DNN model for 2 different atoms.

    Args:: atom: int: A index to distinguish the core atom. (e.g. core atom H -> 1) atom_list: list[int]: A list of
    the index of the neighbor atoms. (e.g. [H, O, O, H, H] -> [1, 2, 2, 1, 1]) embedding_dim: list[int]: Determine
    the structure of the full-connect embedding net structure. (e.g. [20, 40] -> [nn.Linear(3, 20), nn.Linear(20,
    40) ]). linear_layers: list[int]: Determine the structure of the full-connect fitting net structure. Same as
    embedding_dim.

    Input::
    input.size() = (num_neighbor_atoms, 3). Each row is the cartesian position of the atom.

    Output::
    output.size() = (3). Gives out the velocity(displacement) of the core atom.

    Example::
    Please see the example in "__main__".

    """
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
        self.embed12 = nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation, *layers)
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            layers.append(self.activation)
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]), self.activation, *layers)
        self.output_layer = nn.Linear(linear_layers[-1], 3)

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
        return out
    

class DnnSymNew(nn.Module):
    r"""
    Creating a DNN model that is permutation invariant.
    The new version of the DDN_sym model now can be used to fit any atom type number.

    Args:: atom_num: An integer that indicate how many atom types are in the system. atom: int: A index to
    distinguish the core atom, the first index should be 0. (e.g. core atom H -> 0) atom_list: list[int]: A list of
    the index of the neighbor atoms. (e.g. [H, O, O, H, H] -> [0, 1, 1, 0, 0]) embedding_dim: list[int]: Determine
    the structure of the full-connect embedding net structure. (e.g. [20, 40] -> [nn.Linear(3, 20), nn.Linear(20,
    40) ]). linear_layers: list[int]: Determine the structure of the full-connect fitting net structure. Same as
    embedding_dim.

    Input::
    input.size() = (num_neighbor_atoms, 3). Each row is the cartesian position of the atom.

    Output::
    output.size() = (3). Gives out the velocity(displacement) of the core atom.

    Example::
    Please see the example in "__main__".

    """
    def __init__(self, atom_num: int, atom: int, atom_list: list[int], embedding_dim: list[int],
                 linear_layers: list[int]) -> None:
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
                                           module=nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation,
                                                                *layers))
                layers = []
        layers = []
        for i in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            layers.append(self.activation)
        self.linear_layers = nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]), self.activation, *layers)
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
    

class DnnFull(nn.Module):
    def __init__(self, atom_list: list[int], cut_off: float, embedding_dim: list[int], linear_layers: list[int],
                 lattice: torch.Tensor = None) -> None:
        super().__init__()
        self.atom_list = atom_list
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
                                           module=nn.Sequential(nn.Linear(3, embedding_dim[0]), self.activation,
                                                                *layers))
                layers = []
        layers = []
        for j in range(len(linear_layers)-1):
            layers.append(nn.Linear(linear_layers[j], linear_layers[j+1]))
            layers.append(self.activation)
        for i in range(self.atom_num):
            self.linear_list.append(nn.Sequential(nn.Linear(embedding_dim[-1] * 3, linear_layers[0]), self.activation,
                                                  *layers, nn.Linear(linear_layers[-1], 3)))
        
    def forward(self, x):
        # neighboring list generation:
        x_ext = torch.Tensor()
        atom_type = torch.tensor(self.atom_list, dtype=torch.int).repeat(27)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    temp = x + i * self.lattice[0] + j * self.lattice[1] + k * self.lattice[2]
                    x_ext = torch.cat([x_ext, temp], dim=0)
        output = torch.Tensor()
        for n in range(self.atom_num):
            # Neighbor list:
            dist = torch.Tensor()
            x_ext = x_ext - x_ext[n]
            for i in range(len(x_ext)):
                dist = torch.cat([dist, torch.norm(x_ext[i]).expand(1)], dim=0)
            x_n = x_ext[dist <= self.cut_off]
            x_n = torch.cat([x_n[0:n, :], x_n[n+1:, :]], dim=0)
            atom_n = atom_type[dist <= self.cut_off]
            atom_n = torch.cat([atom_n[0:n], atom_n[n+1:]], dim=0)
            # Embedding net:
            g = torch.Tensor()
            for i in range(len(x_n)):
                if atom_type[n] <= atom_n[i]:
                    g_temp = self.embed_dict[f'embed{str(atom_type[n].item())}{str(atom_n[i].item())}'](x_n[i])
                    g = torch.cat([g, g_temp], dim=0)
                else:
                    g_temp = self.embed_dict[f'embed{str(atom_n[i].item())}{str(atom_type[n].item())}'](x_n[i])
                    g = torch.cat([g, g_temp], dim=0)
            g = g.view(len(x_n), -1)
            d = g.T @ x_n
            d = d.view(-1, 1).squeeze()
            out = self.linear_list[n](d)
            output = torch.cat([output, out], dim=0)
        # Normalization
        output = output.view(-1, 3)
        output = output - torch.mean(output, dim=0)
        output = output.view(-1, 1).squeeze()
        return output


def main():
    embed_layers = [20, 40]
    linear_layers = [64, 128, 256]
    atom_list = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    lattice = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    model = DnnFull(atom_list, 5.0, embed_layers, linear_layers, lattice=lattice)
    print(model)
    model.to('cpu')
    input_test = torch.rand(len(atom_list), 3) * 4
    input_test.to('cpu')
    target = torch.rand(3*len(atom_list))
    target.to('cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    output = model(input_test)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    new_input = torch.rand(len(atom_list), 3) * 4
    print(model(new_input))
    # Show that the output is not center of mass drift:
    output = model(new_input)
    print(torch.mean(output.view(-1, 3), dim=0))


if __name__ == "__main__":
    main()
