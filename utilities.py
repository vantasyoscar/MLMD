import numpy as np
import torch
import MLMD

def predict_v(coord,model_list,atoms):
    v_pre = []
    for atom_num in range (atoms):
        model = model_list[atom_num]
        with torch.no_grad():
            output_array = model(torch.from_numpy(coord[atom_num]).float().to("cpu"))
        v_pre.append(output_array.cpu().numpy())    
    return np.array(v_pre)

def step(coord,v):
    return coord + v

def diff(data):
    #输入train data并差分
    v = []
    for i in range(1,len(data)):
        v.append(data[i]-data[i-1])
    return v

def gen_contcar(coord,v,atom_dict,name="CONTCAR"):
    with open(name, 'w') as f:
        f.write("vasp\n")
        f.write("1.0\n")
        for i in (list(atom_dict.values())[0].lattice):
            f.write(str(i)[1:-1]+"\n")
        for (i,j) in atom_dict.items():
            f.write(j.name + " ")
        f.write("\n")
        for (i,j) in atom_dict.items():
            f.write(str(j.number) + " ")
        f.write("\n")
        f.write("C\n")
        for i in coord:
            f.write(str(i)[1:-1]+"\n")
        f.write("\n")
        for i in v:
            f.write(str(i)[1:-1]+"\n")

def get_train_data(atom_dict):
    train_data = []
    n = 0
    for i in atom_dict.values():
        if n == 0:
            for j in i.position:
                train_data.append(j)
            n += 1
        else:
            for j in range(len(train_data)):
                train_data[j] = np.vstack([train_data[j],i.position[j]])
    return train_data

def get_force_from_v(v,atom_dict,atomic_mass):
    for i in range(len(v)):
        cursor = 0
        for j,k in enumerate(atom_dict.values()):            
            v[i][cursor: cursor + k.number] *= atomic_mass[j]
            cursor += k.number
    return v

def gen_xdatcar(coord,atom_dict,name = "XDATCAR"):
    with open('XDATCAR', 'w') as f:
        lattice_inv = np.mat(list(atom_dict.values())[0].lattice).I
        f.write("vasp\n")
        f.write("1.0\n")
        for i in (list(atom_dict.values())[0].lattice):
            f.write(str(i)[1:-1]+"\n")
        for (i,j) in atom_dict.items():
            f.write(j.name + " ")
        f.write("\n")
        for (i,j) in atom_dict.items():
            f.write(str(j.number) + " ") 
        f.write("\n")
        for i in range(len(coord)):
            f.write("Direct configuration=       "+str(i+1)+"\n")
            for j in np.array(coord[i]):
                f.write(str(j.dot(lattice_inv))[2:-2]+"\n")   
