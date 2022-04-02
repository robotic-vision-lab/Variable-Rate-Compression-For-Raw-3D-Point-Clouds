import numpy as np
import matplotlib.pylab  as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
# from io_util.in_out import get_modelnet_input, normalize_pc_unitball, get_ref_pcc_geo_cnnv2_input

#PCC_GEO_CNNV2_GLOB ='/home/ahmed/Data/pcc_geo_cnn_v2_dense/ModelNet40_200_pc512_oct3_4k/**/*.ply'
PCC_GEO_CNNV2_GLOB ='/Users/mdahmedalmuzaddid/Data/ModelNet40_200_pc512_oct3_4k/**/*.ply'


# train_xyz, val_xyz = get_ref_pcc_geo_cnnv2_input(PCC_GEO_CNNV2_GLOB, k=5000, sampling_strategy='closest')
# #train_xyz, train_scale, train_centroid = normalize_pc_unitball(train_xyz)
# sample = train_xyz[0]

def get_data_from_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    return  data['X'], data['X_quant'], data['label'], data['X_string']


def visualize_raw():
    #source ='/Users/mdahmedalmuzaddid/Data/modelnet40_manually_aligned'
    #paths = glob(join(source, '**', f'*.off'), recursive=True)

    X, X_quant, label, X_string = get_data_from_npz(
        "/Users/mdahmedalmuzaddid/Data/compressed_modelnet_combined_decoder_eval.npz") #compressed_autoregressive.npz

    # pc = pc_mesh.get_sample("mesh_random", n=50000, as_PyntCloud=True)
    cls = 11
    X = X[label==cls]
    X_quant = X_quant[label==cls]
    idx = 5
    sample = X[idx]
    fig = plt.figure()#figsize=figsize
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(sample[:,0], sample[:,1], sample[:,2], marker='.', s=4, alpha=.8)
    # sc = ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], marker='.', s=4, alpha=.6)


    ax = fig.add_subplot(122, projection='3d')
    #c = np.concatenate((np.ones([608],dtype=np.int), 1000*np.ones([len(sample)-608],dtype=np.int)))
    a =  1024
    sample = X_quant[idx, :a] #
    sc = ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], marker='.', s=4, alpha=.6, c='r')
    sample = X_quant[idx, a:]
    sc = ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], marker='.', s=4, alpha=.6, c ='b')


    plt.show()

visualize_raw()