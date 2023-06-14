import numpy as np
import MLMD
import time
import torch
import os

def ML_step(predict_coord,predict_num,vec,atomic_mass_rev):


    predict_neighbor_list = MLMD.neighbor_list.get_neighbor_atoms(predict_coord[-1],cutoff = 10,lattice = LiF["Li"].lattice)
    predict_vlist = []
    #print(predict_coord)
    for i in range(predict_num):      
        predict_neighbor_coord = []
        predict_neighboratom = MLMD.neighbor_list.get_coordinates_from_indices(predict_coord[-1],predict_neighbor_list)
        for j ,k in enumerate(predict_neighboratom):
            #平移到中心           
            vector_3d = np.tile(predict_coord[-1][j],(k.shape[0],1))
            predict_neighbor_coord.append(k-vector_3d + vec[j] ) 
        f_predicted = MLMD.utilities.predict_v([predict_neighbor_coord[l] for l in range(62)],model_list,62) 

        #受力改成坐标
        v_predicted = MLMD.utilities.get_force_from_v(f_predicted,LiF,atomic_mass_rev)
        
        predict_vlist.append(v_predicted)
        predict_coord.append(MLMD.utilities.step(predict_coord[-1],v_predicted))

    return predict_coord,predict_vlist

def ML_step_full(model,coord,predict_num,atom_dict,atomic_mass):
    predict_vlist = []
    #往coord后接续predict_num帧
    for i in range(predict_num):      
        
        v_predicted = MLMD.utilities.predict_v_full(model,coord[-1],atom_dict,atomic_mass) 
        predict_vlist.append(v_predicted[0])
        coord.append(coord[-1] + v_predicted[0])

    return coord,predict_vlist

def QM_step(step_num):
    #使用QM运行step_num帧,并更新coord,f并不断学习
    #修改INCAR文件
    with open('vasprun/INCAR_0', 'r+') as f:
        lines = f.readlines()
        lines[0] = "NSW = " +str(step_num) +"\n"
    with open('vasprun/INCAR', 'w') as f:
        for line in lines:
            f.write(line)
    #os.system('cd vasprun;qsub vasp5.pbs')
    os.system("sh run.sh")
   
def learn_from_QM(model,qmcoord,qmforce,lr_new):
    #更新model
    #Learn from QM
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr_new)
    for i in range(len(qmcoord)-1):
        
        print('{:3d}:'.format(i),end= "")
        model,optimizer,loss = MLMD.step.trainning(model,optimizer,qmcoord[i],qmforce[i])

    return model

def trainning(model,optimizer,coord,f):

    time1 = time.time()
    input_array = torch.from_numpy(coord).float().to("cpu")
    target_array = torch.from_numpy(f.reshape(-1)).float().to("cpu")
    output_array = model(input_array)   

    loss = torch.norm(target_array-output_array)/torch.norm(target_array)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Loss: {:.3e},time:{:.2f}s'.format(loss.item(),time.time()-time1))

    return model,optimizer,loss.item()