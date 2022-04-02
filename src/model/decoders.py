import tensorflow as tf
import numpy as np
import itertools

class FC_Decoder(tf.keras.Model):
    def __init__(self, params):
        super(FC_Decoder, self).__init__()
        self.params = params
        self.n_layers = len(self.params['layer_sizes'])
        self.fc_layers = {}

        for i in range(self.n_layers-1):
            self.fc_layers['fc{}'.format(i)] = tf.keras.layers.Dense(units=self.params['layer_sizes'][i],
                                                                     activation='relu',
                                                                     kernel_regularizer=None)
        #final layer with no activation
        self.fc_layers['output'] = tf.keras.layers.Dense(units=self.params['layer_sizes'][-1],
                                                         activation='linear',
                                                         kernel_regularizer=None)
    def call(self, inputs, training=True):
        x = inputs
        for i in range(self.n_layers-1):
            layer_id = 'fc{}'.format(i)
            x = self.fc_layers[layer_id](x)
        y = self.fc_layers['output'](x)
        x_tilde = tf.reshape(y, [-1, self.params['out_shape'][0], self.params['out_shape'][1]])
        return x_tilde

class FoldingNetDecoder(tf.keras.Model):
    def __init__(self):
        super(FoldingNetDecoder, self).__init__()
        # nn.Conv1d(args.feat_dims + 2, args.feat_dims, 1),
        # nn.ReLU(),
        # nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        # nn.ReLU(),
        # nn.Conv1d(args.feat_dims, 3, 1),
        self.num_out_points = 25**2 #2025
        self.meshgrid = [[-0.9, 0.9, 25], [-0.9, 0.9, 25]]#[[-0.9, 0.9, 45], [-0.9, 0.9, 45]]
        self.bneck_size = 512
        self.grid = self.build_grid()
        self.folding1_layers = {}
        self.folding2_layers = {}
        n_filters = [2*self.bneck_size+2, 2*self.bneck_size, 3]
        #n_filters = [1024, 1024, self.num_out_points*3]

        for i in range(len(n_filters)-1):
            self.folding1_layers['conv{}'.format(i)] = tf.keras.layers.Conv1D(filters=n_filters[i],
                                                                          kernel_size=[1],
                                                                          activation='relu',
                                                                          padding='same',
                                                                          use_bias=True)
        self.folding1_layers['conv{}'.format(2)] = tf.keras.layers.Conv1D(filters=n_filters[-1],
                                                                          kernel_size=[1],
                                                                          padding='same',
                                                                          use_bias=True)
        n_filters = [self.bneck_size + 3, self.bneck_size, 3]
        for i in range(len(n_filters)-1):
            self.folding2_layers['conv{}'.format(i)] = tf.keras.layers.Conv1D(filters=n_filters[i],
                                                                          kernel_size=[1],
                                                                          activation='relu',
                                                                          padding='same',
                                                                          use_bias=True)
        self.folding2_layers['conv{}'.format(2)] = tf.keras.layers.Conv1D(filters=n_filters[-1],
                                                                          kernel_size=[1],
                                                                          padding='same',
                                                                          use_bias=True)
    def call(self, inputs, training=True):
        batch_size = tf.shape(inputs)[0]
        grid = tf.tile(self.grid,[batch_size, 1, 1])
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, [1, self.num_out_points,1])
        x = tf.concat( values=[inputs, grid], axis=-1)

        for i in range(len(self.folding1_layers)):
            layer_id = 'conv{}'.format(i)
            x = self.folding1_layers[layer_id](x)
        x = tf.concat(values=[inputs, x], axis=-1)
        for i in range(len(self.folding2_layers)):
            layer_id = 'conv{}'.format(i)
            x = self.folding2_layers[layer_id](x)

        return x


    def build_grid(self):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        #points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = tf.constant(points, dtype=tf.float32)
        points = tf.expand_dims(points, axis=0)
        return points

class OcupancyNetDecoder(tf.keras.Model):
    def __init__(self):
        super(OcupancyNetDecoder, self).__init__()
        self.grid = self.build_grid()
        self.bneck_size = 256
        self.grid_size = [64, 64, 64]
        n_filters = [self.bneck_size , 128, 1]#n_filters = [self.bneck_size + 3, self.bneck_size, 1]

        self.conv_layers ={}
        for i in range(len(n_filters)):
            self.conv_layers['conv{}'.format(i)] = tf.keras.layers.Conv1D(filters=n_filters[i],
                                                                              kernel_size=[1],
                                                                              activation='elu',
                                                                              padding='same',
                                                                              use_bias=True)
    def call(self, inputs, training=True):
        latent_z, transformation = inputs
        batch_size = tf.shape(latent_z)[0]
        #prepare grid
        grid = tf.tile(self.grid, [batch_size, 1, 1])
        meu, scale = transformation[:,:,0:3], transformation[:,:,3:4]
        grid = tf.divide(tf.subtract(grid, meu), scale)
        ##
        latent_z = tf.expand_dims(latent_z, axis=1)
        latent_z = tf.tile(latent_z, [1, np.prod(self.grid_size), 1])
        x = tf.concat(values=[latent_z, grid], axis=-1)

        # x = tf.expand_dims(inputs, axis=1)
        for i in range(len(self.conv_layers)):
            layer_id = 'conv{}'.format(i)
            x = self.conv_layers[layer_id](x)

        x = tf.reshape(x, [-1, *self.grid_size])
        return x

    def build_grid(self):
        h = 63.
        x = tf.linspace(0., h, 64)
        y = tf.linspace(0., h, 64)
        z = tf.linspace(0., h, 64)
        coor = tf.stack(tf.meshgrid(x, y, z, indexing='ij'), axis=-1)
        coor = tf.reshape(coor, (-1, 3))
        coor = tf.expand_dims(coor, axis=0)
        return coor

class UpConvDecoder(tf.keras.Model):
    def __init__(self):
        super(UpConvDecoder, self).__init__()
        n_filters = [512, 256, 256, 128, 3]
        kernel_sizes = [[2,2], [3,3], [4,5], [5,7], [1,1]] #[[2,2], [3,3], [4,5], [5,7], [1,1]]
        strides = [[2,2], [1,1], [2,3], [2,2], [1,1]] #[[2,2], [1,1], [2,3], [3,3], [1,1]]
        self.conv_layers = {}
        for i in range(len(n_filters)-1):
            self.conv_layers['conv{}'.format(i)] = tf.keras.layers.Conv2DTranspose(n_filters[i],
                                                                                   kernel_size= kernel_sizes[i],
                                                                                   strides=strides[i],
                                                                                   padding='Valid',
                                                                                   activation='relu' )
            self.conv_layers['conv{}'.format(len(n_filters)-1)] = tf.keras.layers.Conv2DTranspose(n_filters[-1],
                                                                                   kernel_size=kernel_sizes[-1],
                                                                                   strides=strides[-1],
                                                                                   padding='Valid',)
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        embedding_size = inputs.shape[1]
        x = tf.reshape(inputs, [batch_size, 1, 2, embedding_size//2]) ##########******
        for i in range(len(self.conv_layers)):
            layer_id = 'conv{}'.format(i)
            x = self.conv_layers[layer_id](x)
        x = tf.reshape(x, [batch_size, -1, 3]) #batch * 1440 * 3
        return x

class SharedFCDecoder(tf.keras.Model):
    def __init__(self):
        super(SharedFCDecoder, self).__init__()
        n_filters = [512, 128, 64, 3]
        self.conv_layers = {}

        for i in range(len(n_filters) - 1):
            self.conv_layers['conv{}'.format(i)] = tf.keras.layers.Conv1D(filters=n_filters[i],
                                                                              kernel_size=[1],
                                                                              activation='relu',
                                                                              padding='same',
                                                                              use_bias=True)
        self.conv_layers['conv{}'.format(2)] = tf.keras.layers.Conv1D(filters=n_filters[-1],
                                                                          kernel_size=[1],
                                                                          padding='same',
                                                                          use_bias=True)

    def call(self, inputs, training=None, mask=None):
        global_code, local_code = inputs
        global_code = tf.tile(tf.expand_dims(global_code, axis=1), (1,local_code.shape[1], 1))
        latent_code = tf.concat((global_code, local_code), axis=-1)
        x = latent_code
        for i in range(len(self.conv_layers)):
            layer_id = 'conv{}'.format(i)
            x = self.conv_layers[layer_id](x)
        return x

class DeltaDecoder(tf.keras.Model):
    def __init__(self, params):
        super(DeltaDecoder, self).__init__()
        self.params = params
        layer_size = [1024, 1024, 1792 *3]
        self.n_layers = len(layer_size)
        self.fc_layers = {}

        for i in range(self.n_layers-1):
            self.fc_layers['fc{}'.format(i)] = tf.keras.layers.Dense(units=layer_size[i],
                                                                     activation='relu',
                                                                     kernel_regularizer=None)
        #final layer with no activation
        self.fc_layers['output'] = tf.keras.layers.Dense(units=layer_size[-1],
                                                         activation='linear',
                                                         kernel_regularizer=None)

    def call(self, inputs, training=True):
        x = inputs
        x1 = inputs[:, :256*3]
        x2 = inputs[:, 256*3:]
        for i in range(self.n_layers-1):
            layer_id = 'fc{}'.format(i)
            x = self.fc_layers[layer_id](x)
        y = self.fc_layers['output'](x)
        x1_extend = tf.tile(x1, (1,7))
        x_tilde = tf.concat((x1_extend+y, x1), axis=-1)
        x_tilde = tf.reshape(x_tilde, [-1, self.params['out_shape'][0], self.params['out_shape'][1]])
        return x_tilde


