#!/home/cwj/opt/anaconda3/bin/python

import numpy as np
import sys
import torch
import torch.nn as nn
sys.path.append('..')
import MLMD
import matplotlib.pyplot as plt
from MLMD.step import QM_step,ML_step_full
import os

#需要自定义输入的参数

total_step = 100
initial_step = 25
step_ml = 5
step_qm = 5
tol = 0.4
atom_mass = [19,6.941]
embed_layers = [16,32,32,64]
linear_layers = [64,128,256]
lr = 0.004

print("Initial QM step")
QM_step(initial_step)
#读取XDATCAR
LiF = MLMD.atom_io.loadfile(1,"vasprun/XDATCAR")
traj = MLMD.utilities.get_train_data(LiF)
lattice  = torch.tensor(list(LiF.values())[0].lattice).float()
#制作原子类型表（对网络重要）
atom_list = MLMD.neighbor_list.get_atom_list(LiF)
#制作测试集
train_data = MLMD.utilities.get_train_data(LiF)
v = MLMD.utilities.diff(train_data)
f = MLMD.utilities.get_force_from_v(v,LiF,atom_mass)
#初始化模型
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLMD.model.DnnFull(atom_list,5,embed_layers,linear_layers,lattice).to(device)

cirtlist = []
#初始模型的训练
print("Starting to train the initial step")
model = MLMD.step.learn_from_QM(model,train_data,f,lr)

while len(traj) < total_step:
    
    #使用初始模型预测受力，更新位置坐标
    print("starting ML step")
    predict_coord,predict_vlist = ML_step_full(model,[traj[-1]],step_ml,LiF,atom_mass)

    #将最后一步输出至CONTCAR，进行QM运算
    MLMD.utilities.gen_contcar(predict_coord[step_ml],predict_vlist[step_ml-1],LiF,"vasprun/POSCAR")

    print("starting QM step")
    QM_step(step_qm)

    #查看QM和ML的误差
    LiF = MLMD.atom_io.loadfile(1,"vasprun/XDATCAR")
    train_data = MLMD.utilities.get_train_data(LiF)
    v = MLMD.utilities.diff(train_data)
    f = MLMD.utilities.get_force_from_v(v,LiF,atom_mass)
    test_v_predicted = MLMD.utilities.predict_v_full(model,train_data[0],LiF,atom_mass)
    crit = np.linalg.norm(test_v_predicted - v[0])/np.linalg.norm(v[0])
    cirtlist.append(crit)
    print("Percentage bias for traj",len(traj),":",crit)
    #如果小于容忍值，更新坐标（还没写else，先不管）
    tol = 0.3
    print(len(traj),len(predict_coord),len(train_data))
    if crit < tol:
        print("Recieve ML step")
        traj += predict_coord[1:]
        traj += train_data       
        #符合要求的同时进行学习，后续模型的训练采用更小的学习率
        model = MLMD.step.learn_from_QM(model,train_data,f,0.001)
        #导出XDATCAR
        MLMD.utilities.gen_xdatcar(traj,LiF)
    else:
        #不更新轨迹并重新训练
        print("Reject ML step")
        model = MLMD.step.learn_from_QM(model,train_data,f,0.001)