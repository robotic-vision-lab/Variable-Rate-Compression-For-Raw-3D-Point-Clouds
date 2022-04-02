import os, sys
import numpy as np
import tensorflow as tf
from io_util.in_out import get_modelnet_input, normalize_pc_unitball, \
    get_ref_pcc_geo_cnnv2_input, get_local_pca_feature
from model.encoder_decoder import AutoEncoder
from util.params_manager import params
from datetime import datetime
from util.general_utils import apply_augmentations_single_pointcloud, apply_augmentations
import sklearn
from io_util import in_out


EXPERIMENT_NAME = "multi_decoder_evaluation"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "../logs",  EXPERIMENT_NAME, datetime.now().strftime("%m%d-%H%M") )

DATA_DIR = '/home/ahmed/Data/pc_compress_evaluation/pointnet/classification'
out_filename = 'out_' + EXPERIMENT_NAME + '.txt'
SEED = 42
#CHECKPOINT_PATH = '/home/ahmed/Data/pc_compress_evaluation/my_model/autoregressive/fc_decoder_with_weighted_loss/training_checkpoints'

def augment_input(pc_xyz, scale_factor, train_params):
    if train_params['z_rotate']:
        augmented_xyz = [pc_xyz]
        for i in range(2):
            augmented_xyz.append(apply_augmentations(pc_xyz, z_rotate=True))

        pc_xyz = np.concatenate(augmented_xyz, axis=0)
        if scale_factor:
            scale_factor = np.tile(scale_factor, (3,1,1))
    if scale_factor:
        pc_xyz, scale_factor = sklearn.utils.shuffle(pc_xyz, scale_factor, random_state=SEED)
    else:  pc_xyz= sklearn.utils.shuffle(pc_xyz, random_state=SEED)
    return pc_xyz, scale_factor

def augment_input_with_label(train_params, pc_xyz, label=None ):
    if train_params['z_rotate']:
        augmented_xyz = [pc_xyz]
        for i in range(2):
            augmented_xyz.append(apply_augmentations(pc_xyz, z_rotate=True))
        pc_xyz = np.concatenate(augmented_xyz, axis=0)
        if label is not None:
            label = np.tile(label, (3))
    if label is not None:
        pc_xyz, label = sklearn.utils.shuffle(pc_xyz, label, random_state=SEED)
    else:  pc_xyz= sklearn.utils.shuffle(pc_xyz, random_state=SEED)
    return pc_xyz, label

def create_model():
    inputs = tf.keras.Input(shape=(2048, 3))
    transformation = tf.keras.Input(shape=(1,4), )
    autoencoder = AutoEncoder(params)
    x_tilde = autoencoder([inputs, transformation])
    model = tf.keras.Model(inputs=[inputs, transformation], outputs=x_tilde)
    return model

def get_data_from_npz(DATA_DIR):
    train_data = in_out.read_from_numpy(os.path.join(DATA_DIR,  'train_data.npz')) #shapeNet_seg_train.npz,'train_data.npz'
    test_data = in_out.read_from_numpy(os.path.join(DATA_DIR,  'test_data.npz' )) #'shapeNet_seg_val.npz' # 'test_data.npz'
    return train_data[0], train_data[1], test_data[0], test_data[1] #train_xyz, test_xyz

def get_train_test_data(train_params):

    train_xyz, train_label, test_xyz, test_label = get_data_from_npz(DATA_DIR)#SEG_NPZ_DATA_DIR #get_modelnet_input(MODELNET40_TOPDIR)
    #comment out if you don't to normalize pc
    train_xyz, train_transformation = normalize_pc_unitball(train_xyz)
    test_xyz, test_transformation = normalize_pc_unitball(test_xyz)

    train_xyz, train_label = augment_input_with_label(train_params, train_xyz, train_label, )
    return train_xyz, train_label, test_xyz, test_label

def train():
    #region setup inputs
    train_xyz, train_label, test_xyz, test_label = get_train_test_data(train_params = params['train_params'])
    n_train = len(train_xyz)
    n_test = len(test_xyz)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_xyz, train_label)).batch(
        params['train_params']['batch_size'],
        drop_remainder=True).shuffle(len(train_xyz))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_xyz,test_label)).batch(
        params['train_params']['batch_size'],
        drop_remainder=True)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_data_init_op = iter.make_initializer(train_dataset)
    test_data_init_op = iter.make_initializer(test_dataset)
    xyz, label = iter.get_next()

    input_placeholder = tf.keras.Input(shape=(2048, 3))
    label_placeholder = tf.keras.Input(shape=(1,), )
    autoencoder = AutoEncoder(params)
    x_tilde = autoencoder([xyz, label])
    loss = autoencoder.calc_loss(xyz, x_tilde)
    metric = autoencoder.metrics
    train_step = autoencoder.setup_optimizer(loss)

    train_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'))
    test_summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
    tf.summary.histogram('Embedding', autoencoder.embedding)
    tf.summary.scalar('chamfer loss', loss)
    merged = tf.summary.merge_all()
    checkpoint_directory = os.path.join(LOG_DIR, "training_checkpoints")
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    '''For continuing traning from previous checkpoint'''
    # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    # ckpt = tf.train.Checkpoint(step=tf.Variable(1),  net=autoencoder)
    # status = ckpt.restore(tf.train.latest_checkpoint(checkpoint_directory))
    embedding_dict = {}
    with tf.Session() as sess:
        if 'CHECKPOINT_PATH' in globals():
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        else:
            sess.run(tf.global_variables_initializer())
            os.makedirs(checkpoint_directory)

        #data_iter_init_op = iter.make_initializer(train_dataset)
        for epoch in range(params['train_params']['mx_epochs']):
            epoch_embed = []
            ######### Training
            sess.run(train_data_init_op)
            train_chamfer_loss, train_entropy_loss = 0, 0
            for iteration in range(n_train // params['train_params']['batch_size']):
                xyz1, label1 = sess.run([xyz, label])
                feed_dict = { xyz: xyz1, label: label1}
                #summary, _, c_l, e_l = sess.run([merged, train_step, autoencoder.chamfer_loss, autoencoder.entropy_loss ], feed_dict )
                embed, summary, _, c_l, e_l = sess.run(
                    [autoencoder.embedding, merged, train_step, autoencoder.chamfer_loss, autoencoder.entropy_loss], feed_dict)
                #summary,c_l, e_l = sess.run([merged, autoencoder.chamfer_loss, autoencoder.entropy_loss ], feed_dict )
                epoch_embed.append(embed)
                train_chamfer_loss += c_l
                train_entropy_loss += e_l
            train_chamfer_loss = train_chamfer_loss / (iteration + 1)
            train_entropy_loss = train_entropy_loss / (iteration + 1)
            train_summary_writer.add_summary(summary, epoch)
            train_summary_writer.flush()

            ########## Testing
            sess.run(test_data_init_op)
            test_chamfer_loss, test_entropy_loss = 0, 0
            for iteration in range(n_test // params['train_params']['batch_size']):
                summary, c_l, e_l = sess.run([merged, autoencoder.chamfer_loss, autoencoder.entropy_loss ], )
                test_chamfer_loss += c_l
                test_entropy_loss += e_l
            test_chamfer_loss = test_chamfer_loss / (iteration + 1)
            test_entropy_loss = test_entropy_loss / (iteration + 1)

            test_summary_writer.add_summary(summary, epoch)
            test_summary_writer.flush()

            ############ saving weight
            if epoch % 50 == 0:
                embedding_dict[epoch] = np.reshape(epoch_embed, [-1, 1024])
                np.save(os.path.join(LOG_DIR,'wighted_embedding.npy'), embedding_dict)

            if epoch%1 == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_directory,"model.ckpt"), global_step=epoch)
            txt = 'Epoch:{}\tTrain chamfer loss:{:.4}\tTrain entropy loss:{:.4}\t\tTest chamfer loss:{:.4}\tTest entropy loss:{:.4}'\
                .format(epoch, train_chamfer_loss, train_entropy_loss, test_chamfer_loss, test_entropy_loss)
            print(txt)
            f = open(out_filename, "a+")
            f.write(txt+'\r\n')
            f.close()


train()
