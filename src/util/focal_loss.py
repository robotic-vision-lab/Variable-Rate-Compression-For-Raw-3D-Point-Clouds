import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(y_true, y_pred, gamma=2, alpha=0.9):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    #return -K.sum(occupancy_label*alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(occupancy_label*(1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return -K.sum(  alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def distance_matrix(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances

def av_dist_sum(arrays):
    """
    arguments:
        arrays: array1, array2
    returns:
        sum of av_dist(array1, array2) and av_dist(array2, array1)
    """
    array1, array2 = arrays
    array2 = tf.where(array2>=0.0)
    if tf.math.greater(tf.shape(array2)[0],  0) and tf.math.less(tf.shape(array2)[0], 10000) :
        av_dist1 = av_dist(array1, array2)
        av_dist2 = av_dist(array2, array1)
        return av_dist1+av_dist2
    return 0.0

def chamfer_distance_tf(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = tf.reduce_mean(
               tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float64)
           )
    return tf.reduce_mean(dist)
