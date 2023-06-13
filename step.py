import numpy as np
import MLMD

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