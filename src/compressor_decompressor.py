import os, sys
import numpy as np
import tensorflow as tf
from io_util.in_out import get_modelnet_input, normalize_pc_unitball, \
    get_ref_pcc_geo_cnnv2_input, get_local_pca_feature
from model.encoder_decoder import AutoEncoder
from util.params_manager import params
from io_util import in_out
from timeit import default_timer as timer


TRAINED_MODEL_DIR = "/home/ahmed/Data/pc_compress_evaluation/my_model/freq_fet_double_decoder/double_decoder_15_28_chamfer/training_checkpoints"
COMPRESSION_STORE_BASEDIR =  "/home/ahmed/Data/pc_compress_evaluation/compressed_data"
NPZ_DATA_DIR = '/home/ahmed/Data/pc_compress_evaluation/pointnet/classification'
VEC_LENGTH = 1024

def save_pointcloud_npz(X, X_quant, label, X_string, file_name):
    np.savez(file_name, X=X, X_quant= X_quant, label=label, X_string=X_string)

def create_store_dir(store_dir):
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

def get_data_from_npz(DATA_DIR):
    train_data = in_out.read_from_numpy(os.path.join(DATA_DIR, 'train_data.npz' ))
    test_data = in_out.read_from_numpy(os.path.join(DATA_DIR, 'test_data.npz'))
    return train_data[0], train_data[1], test_data[0], test_data[1] #train_xyz, test_xyz

def get_compressed_data_from_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    return  data['X'], data['X_quant'], data['label'], data['X_string']

def compress():
    create_store_dir(COMPRESSION_STORE_BASEDIR)
    train_params = params['train_params']

    train_xyz, train_label,test_xyz, test_label,    = get_data_from_npz(NPZ_DATA_DIR)
    n_poincloud = len(train_xyz)
    n_train = len(train_xyz) #int(n_poincloud*0.9)
    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices((train_xyz, train_label))\
        .batch(batch_size, drop_remainder=True)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_data_init_op = iter.make_initializer(train_dataset)
    #test_data_init_op = iter.make_initializer(test_dataset)

    xyz, label = iter.get_next()
    autoencoder = AutoEncoder(params)
    x_tilde = autoencoder([xyz, label], training=False)

    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        else:
            print("Can not proceed without a trained model")
            return
        sess.run(train_data_init_op)
        # sess.run(test_data_init_op)

        X_quant_list = []
        quant_string_list = []
        timing_list = []

        for iteration in range(n_train // batch_size):
            #X = sess.run(xyz )
            start = timer()
            X, X_quant, quant_string = sess.run([xyz, x_tilde, autoencoder.embedding_string] )
            #quant_string = sess.run( autoencoder.embedding_string)
            end = timer()
            X_quant_list.append(X_quant)
            quant_string_list.append(quant_string)
            t = end - start
            if 5 < iteration < 1005 :timing_list.append(t) #warm start

        #print("Compression time Mean {}\t variation: {}".format( np.mean(timing_list), np.var(timing_list)))
        print("Compression time Mean {}\t std: {}".format( 1000*np.mean(timing_list)/batch_size, 1000*np.std(timing_list)/batch_size))

    X_quant_list = np.concatenate(X_quant_list)
    quant_string_list = np.concatenate(quant_string_list)
    n = len(X_quant_list)
    save_pointcloud_npz(train_xyz[:n], X_quant_list, train_label[:n], quant_string_list,
                    file_name=os.path.join(COMPRESSION_STORE_BASEDIR,"compressed_modelnet_combined_decoder_eval.npz") )

def decompress():
    _,_,_, encoded_string = get_compressed_data_from_npz("/home/ahmed/Data/pc_compress_evaluation/compressed_data/compressed_shapenet_seg_val_chamfer-7.npz")
    """Initializes the decompression model"""
    xyz = tf.keras.Input(shape=(2048, 3))
    label = tf.keras.Input(shape=(1,), )
    comp_string = tf.placeholder(tf.string)
    autoencoder = AutoEncoder(params)
    x_tilde = autoencoder([xyz, label], False)
    #x_tilde = autoencoder.decompress(comp_string)
    #
    # autoencoder = AutoEncoder(params, )
    # x_tilde = autoencoder(xyz, training=False)


    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    batch_sizse = 32
    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(TRAINED_MODEL_DIR)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
            print("Trained model restored")
        else:
            print("Can not proceed without a trained model")
            return
        timing_list = []
        for i in range(len(encoded_string)//batch_sizse):

            start = timer()
            xyz_reconstructed = sess.run(x_tilde, feed_dict={autoencoder.embedding_string:encoded_string[(i-1)*batch_sizse:i*batch_sizse]})
            end = timer()
            t = end - start
            if 5< i < 1005:timing_list.append(t)
        print("Deompression time Mean {}\t std: {}".format( 1000*np.mean(timing_list)/batch_sizse, 1000*np.std(timing_list)/batch_sizse))

        print(np.shape(xyz_reconstructed))

#compress()
#decompress()
if __name__ == "__main__":
    if len(sys.argv)< 2:
        print("Missing Compress or Decompress argument.")
    if len(sys.argv)> 2:
        VEC_LENGTH = int(sys.argv[2])
    if sys.argv[1] == 'compress':
        compress()
    elif sys.argv[1] == 'decompress':
        decompress()
