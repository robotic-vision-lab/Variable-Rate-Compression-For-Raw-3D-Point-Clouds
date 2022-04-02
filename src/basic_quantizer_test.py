import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

#tf.enable_eager_execution()

dtype = tf.float32
BATCH_SIZE = 64
SAMPLE_SIZE = 10000
EPS = 1e-10

np.random.seed(seed=1234)
X = np.random.multivariate_normal(mean=[.2,.7], cov=.01*np.identity(2), size= (SAMPLE_SIZE), ).astype(np.float32)
X = tf.data.Dataset.from_tensor_slices(X, ).batch(BATCH_SIZE).shuffle(SAMPLE_SIZE)

def get_quantization_level(range=[0., 1.], num_level=16):
    d = (range[1]-range[0])/(num_level)
    levels = tf.Variable(initial_value=tf.range(range[0], range[1], delta=d, dtype=tf.float32), trainable=True)
    return levels


centers = get_quantization_level()#tf.Variable(initial_value=tf.range(0., 1., delta = .16,dtype=dtype), trainable= True)

def qunatize(X, centers):
    _HARD_SIGMA = 1e7
    num_centers = tf.shape(centers)[0]

    X = tf.expand_dims(X, axis=-1)
    dist = tf.square(X - centers)
    phi_soft = tf.nn.softmax(-1 * dist, axis=-1)
    phi_hard = tf.nn.softmax(-_HARD_SIGMA * dist, axis=-1)
    symbols_hard = tf.argmax(phi_hard, axis=-1)
    phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=dtype)
    soft_out = tf.matmul(phi_soft, tf.expand_dims(centers, axis=-1))
    hard_out = tf.matmul(phi_hard,tf.expand_dims(centers, axis=-1))

    z = soft_out + tf.stop_gradient(hard_out - soft_out)
    z = tf.squeeze(z, axis=-1)

    return z, phi_soft, phi_hard


def entroy_loss(phi_soft, phi_hard):
    meu_soft_phi = tf.reduce_mean(phi_soft, axis=[0,1])
    soft_en_loss = -tf.reduce_sum(meu_soft_phi*tf.math.log(meu_soft_phi))/tf.cast(tf.math.log(2.),dtype=dtype)
    meu_hard_phi = tf.reduce_mean(phi_hard, axis=[0, 1])
    meu_hard_phi += EPS
    hard_en_loss = -tf.reduce_sum(meu_hard_phi * tf.math.log(meu_hard_phi)) / tf.cast(tf.math.log(2.), dtype=dtype)
    return soft_en_loss, hard_en_loss


def train():
    epochs = 5000
    optimizer = tf.keras.optimizers.Adam(learning_rate=.0002)
    for epoch in range(epochs):
        #print("\nStart of epoch %d" % (epoch,))
        epoch_reconstruction_loss = 0
        epoch_soft_entropy_loss = 0
        epoch_hard_entropy_loss = 0
        for step, x in enumerate(X):
            with tf.GradientTape() as tape:
                z, phi_soft, phi_hard = qunatize(x,centers)
                soft_en_loss, hard_en_loss = entroy_loss(phi_soft, phi_hard)
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.MAE( x, z)) #tf.reduce_mean(tf.square(x - z))
                total_loss = reconstruction_loss + .02*soft_en_loss
                grads = tape.gradient(total_loss, [centers])
                optimizer.apply_gradients(zip(grads, [centers]))
                epoch_soft_entropy_loss +=soft_en_loss
                epoch_hard_entropy_loss +=hard_en_loss
                epoch_reconstruction_loss +=  reconstruction_loss
        if epoch %20 == 0:
            epoch_soft_entropy_loss /= (step+1)
            epoch_hard_entropy_loss /= (step + 1)
            epoch_reconstruction_loss /= (step+1)

            print(centers.numpy(), "Epoch", epoch,
                  '\tLoss:',epoch_reconstruction_loss.numpy(),
                  '\tSen_loss:', epoch_soft_entropy_loss.numpy(),
                  '\tHen_loss:', epoch_hard_entropy_loss.numpy())

def calc_chamfer_loss(x_qaunt, x_tilde):

    batch_size = np.shape(x_qaunt)[0]
    chamfer_dist =[]
    for i in range(batch_size):
        x = x_qaunt[i]
        y = x_tilde[i]
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=10, algorithm='kd_tree',).fit(x)
        temp = x_nn.kneighbors(y)
        min_y_to_x = temp[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=10, algorithm='kd_tree', ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist.append( np.mean(np.square(min_y_to_x)) + np.mean(np.square(min_x_to_y)))
    return np.mean(chamfer_dist)


def test_chamfer_loss():
    from external.structural_losses.tf_nndistance import nn_distance
    X = [[5,2,0],[4,5,0]]
    X_tilde = [[2, 2, 0], [4, 6, 0]]
    X = np.expand_dims(X,axis=0)

    X_tilde = np.expand_dims(X_tilde, axis=0)
    loss1 = calc_chamfer_loss(X, X_tilde)

    # X = tf.constant(X,dtype=float)
    # X_tilde = tf.constant(X_tilde,dtype=float)
    # cost_p1_p2, _, cost_p2_p1, _ = nn_distance(X_tilde, X)
    # loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

    #print("loss:{}\tloss1:{}".format(loss,loss1))

def test_nd_scatter():
    a = tf.reshape(tf.tile(tf.expand_dims(tf.transpose(tf.range(32)), -1), [1, 2048]), shape=[-1,1])

    print(a.numpy())
    # indices = tf.constant([[0,0,1], [0,1,1],[1,0,0], [1,0,1]])
    # updates = tf.constant([1,1,1,1])
    # shape = tf.constant([2, 2, 2])
    # scatter = tf.scatter_nd(indices, updates, shape)
    # print(scatter.numpy())
test_nd_scatter()


#test_chamfer_loss()
train()




