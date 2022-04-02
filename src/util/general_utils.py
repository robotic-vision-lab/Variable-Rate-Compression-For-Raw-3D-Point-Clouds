'''
Created on November 26, 2017

@author: optas
'''

import numpy as np
from numpy.linalg import norm
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


def rand_rotation_matrix(deflection=1.0, seed=None):
    '''Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.

    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M.astype(np.float32)


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

        
def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud

def apply_augmentations_single_pointcloud(pointcloud):

    r_rotation = rand_rotation_matrix()
    r_rotation[0, 2] = 0
    r_rotation[2, 0] = 0
    r_rotation[1, 2] = 0
    r_rotation[2, 1] = 0
    r_rotation[2, 2] = 1
    #pointcloud = pointcloud.dot(r_rotation)
    pointcloud = tf.linalg.matmul(pointcloud, r_rotation)
    return pointcloud

def apply_augmentations(batch, gauss_augment= False, z_rotate = False):
    res_batch = batch
    if gauss_augment  or z_rotate:
        res_batch = batch.copy()

    if gauss_augment:
        mu = 0.#conf.gauss_augment['mu']
        sigma = 0.02#conf.gauss_augment['sigma']
        res_batch += np.random.normal(mu, sigma, batch.shape)

    if z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        res_batch = res_batch.dot(r_rotation)
    return res_batch

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def calc_chamfer_loss(x_qaunt, x_tilde):

    batch_size = np.shape(x_qaunt)[0]
    chamfer_dist =[]
    for i in range(batch_size):
        x = x_qaunt[i]
        y = x_tilde[i]
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        #chamfer_dist.append( np.mean(min_y_to_x) + np.mean(min_x_to_y))
        chamfer_dist.append(np.mean(np.square(min_y_to_x)) + np.mean(np.square(min_x_to_y)))
    return np.mean(chamfer_dist)


# def apply_augmentations(batch, conf):
#     if conf.gauss_augment is not None or conf.z_rotate:
#         batch = batch.copy()
#
#     if conf.gauss_augment is not None:
#         mu = conf.gauss_augment['mu']
#         sigma = conf.gauss_augment['sigma']
#         batch += np.random.normal(mu, sigma, batch.shape)
#
#     if conf.z_rotate:
#         r_rotation = rand_rotation_matrix()
#         r_rotation[0, 2] = 0
#         r_rotation[2, 0] = 0
#         r_rotation[1, 2] = 0
#         r_rotation[2, 1] = 0
#         r_rotation[2, 2] = 1
#         batch = batch.dot(r_rotation)
#     return batch
