import six
import warnings
import numpy as np
import os
import os.path as osp
import re
from six.moves import cPickle
from multiprocessing import Pool
from .pc_io import get_files as pc_io_get_files, load_points as pc_io_load_points

from  util.general_utils import rand_rotation_matrix
from  external.python_plyfile.plyfile import PlyElement, PlyData
from sklearn.neighbors import NearestNeighbors

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}


def snc_category_to_synth_id():
    d = snc_synth_id_to_category
    inv_map = {v: k for k, v in six.iteritems(d)}
    return inv_map


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    size = cPickle.load(inFile)
    for _ in range(size):
        yield cPickle.load(inFile)
    inFile.close()


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme: 
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id


def load_all_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply', verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]
    pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, verbose=verbose)
    return pclouds
    #return PointCloudDataSet(pclouds, labels=syn_ids + '_' + model_ids, init_shuffle=False)


def load_point_clouds_from_filenames(file_names, n_threads, loader, verbose=False):
    pc = loader(file_names[0])[0]
    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds), len(np.unique(class_ids))))

    return pclouds, model_names, class_ids


class PointCloudDataSet(object):
    '''
    See https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py
    '''

    def __init__(self, point_clouds, noise=None, labels=None, copy=True, init_shuffle=True):
        '''Construct a DataSet.
        Args:
            init_shuffle, shuffle data before first epoch has been reached.
        Output:
            original_pclouds, labels, (None or Feed) # TODO Rename
        '''

        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels

        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)

        if noise is not None:
            assert (type(noise) is np.ndarray)
            if copy:
                self.noisy_point_clouds = noise.copy()
            else:
                self.noisy_point_clouds = noise
        else:
            self.noisy_point_clouds = None

        if copy:
            self.point_clouds = point_clouds.copy()
        else:
            self.point_clouds = point_clouds

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]
        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = self.noisy_point_clouds[perm]
        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.noisy_point_clouds is None:
            return self.point_clouds[start:end], self.labels[start:end], None
        else:
            return self.point_clouds[start:end], self.labels[start:end], self.noisy_point_clouds[start:end]

    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)
        pc = self.point_clouds[perm]
        lb = self.labels[perm]
        ns = None
        if self.noisy_point_clouds is not None:
            ns = self.noisy_point_clouds[perm]
        return pc, lb, ns

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = np.vstack((self.point_clouds, other_data_set.point_clouds))

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)

        if self.noisy_point_clouds is not None:
            self.noisy_point_clouds = np.vstack((self.noisy_point_clouds, other_data_set.noisy_point_clouds))

        self.num_examples = self.point_clouds.shape[0]

        return self


def normalize_pc_unitball(pc_dataset):
    '''

    :param pc_dataset:
    :return: transformation: [#pc, 1, 4]
    '''
    centroid = np.mean(pc_dataset, axis=1, keepdims=True)
    pc_dataset = pc_dataset - centroid
    scale_factor = np.max(np.sqrt(np.sum(pc_dataset**2, axis=2, keepdims=True)), axis=1, keepdims=True)
    pc_dataset /= scale_factor
    transformation = np.concatenate((scale_factor, centroid), axis=-1)
    return pc_dataset, transformation


# def normalize_pc_unitball1(pc_dataset):
#     n_sample = len(pc_dataset)
#     scale_factor = np.zeros(shape=(n_sample,), dtype=np.float32)
#     for i in range(n_sample):
#         pc = pc_dataset[i]
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#         pc = pc / m
#         scale_factor[i] = m
#         pc_dataset[i] = pc
#     return pc_dataset, scale_factor
def sample_random_point(points, k):
    sample_points = np.zeros([len(points),k,3],dtype=np.float32)
    for i in range(len(points)):
        x = points[i]
        if len(x)>=k:
            p_idx = np.random.choice(x.shape[0], k, replace=False)
            x = x[p_idx, :]
        else:
            x =np.concatenate((x,np.zeros([k-len(x),3])),axis=0)
        sample_points[i] =x
    return sample_points

def sample_closest_point(points, k):
    sample_points = np.zeros([len(points), k, 3], dtype=np.float32)
    for i in range(len(points)):
        x = points[i]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(x)
        if len(x) >= k:
            _, p_idx = nbrs.kneighbors(x[0:1])
            x = x[p_idx[0], :]
        else:
            x = np.concatenate((x, np.zeros([k - len(x), 3])), axis=0)
        sample_points[i] = x
    return sample_points


def get_modelnet_input(MODELNET40_TOPDIR):
    pc_xyz = load_all_point_clouds_under_folder(MODELNET40_TOPDIR,
                                                n_threads=8,
                                                file_ending='.ply',
                                                verbose=True)
    return pc_xyz


def get_ref_pcc_geo_cnnv2_input(PCC_GEO_CNNV2_GLOB, k=2048, sampling_strategy= 'random'):
    files =  pc_io_get_files(PCC_GEO_CNNV2_GLOB)
    assert len(files) > 0
    points = pc_io_load_points(files)
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
    points_train = points[files_cat == 'train']
    points_val = points[files_cat == 'test']
    if sampling_strategy == 'random':
        sampler = sample_random_point
    elif sampling_strategy == 'closest':
        sampler = sample_closest_point
    else:return points_train, points_val


    points_train = np.asarray(sampler(points_train, k))
    points_val = sampler(points_val, k)
    # pc_xyz, scale_factor = sklearn.utils.shuffle(pc_xyz, scale_factor, random_state=SEED)
    return points_train, points_val

from sklearn.decomposition import PCA
from multiprocessing import Pool
def get_feature(pc, n_neighbors=8):
    pca_feature = np.zeros_like(pc)
    x_nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree', metric='l2').fit(pc)  # leaf_size=1
    dis, idx = x_nn.kneighbors(pc)
    pca = PCA(n_components=3)
    for i, pidx in enumerate(idx):
        x = pc[pidx, :]
        pca.fit(x)
        normal = pca.components_[2]
        pca_feature[i] = normal

    return pca_feature

def get_local_pca_feature(batch_pc, n_neighbors=8):
    with Pool() as pool:
        pca_feature = pool.map(get_feature, batch_pc)
    pca_feature = np.asarray(pca_feature)
    return pca_feature

def read_from_numpy(file_name):
    npzfile = np.load(file_name)
    return npzfile['X'], npzfile['label']

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc



