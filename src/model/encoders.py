import tensorflow as tf
import sys, os
from util.pointnet_util import  sample_and_group_all, sample_and_group

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point


class PointNetEncoder(tf.keras.Model):
    def __init__(self, params,):
        super(PointNetEncoder, self).__init__()
        self.params = params
        self.n_layers = len(params['n_filters'])
        self.conv_layers = {}

        for i in range(self.n_layers):
            self.conv_layers['conv{}'.format(i)] = tf.keras.layers.Conv1D(  filters=self.params['n_filters'][i],
                                                                            kernel_size=[1],
                                                                            activation='relu',
                                                                            padding=self.params['padding'],
                                                                            data_format='channels_last',
                                                                            kernel_regularizer = None,
                                                                            use_bias=True)

    def call(self, inputs, training=True):
        x = inputs
        for i in range(self.n_layers):
            layer_id = 'conv{}'.format(i)
            x = self.conv_layers[layer_id](x)
        if self.params['symmetry']:
            x = tf.reduce_max(x, axis=1)
        #x = tf.keras.layers.Reshape([x.shape[1]*x.shape[2]])(x)
        return x


class PointNetPlusSetAbstactionMSG(tf.keras.Model):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
           Input:
               xyz: (batch_size, ndataset, 3) TF tensor
               points: (batch_size, ndataset, channel) TF tensor
               npoint: int32 -- #points sampled in farthest point sampling
               radius: list of float32 -- search radius in local region
               nsample: list of int32 -- how many points in each local region
               mlp: list of list of int32 -- output size for MLP on each point
               use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
               use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
           Return:
               new_xyz: (batch_size, npoint, 3) TF tensor
               new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
       '''
    def __init__(self, npoint, radius_list,
                 nsample_list, mlp_list,
                 is_training, use_xyz=True,
                 use_nchw=False):
        super(PointNetPlusSetAbstactionMSG, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = mlp_list
        self.use_xyz=use_xyz
        self.use_nchw = use_nchw
        self.conv_layers ={}

        for i in range(len(self.radius_list)):
            layer_for_cur_radius = {}
            for j, num_out_channel in enumerate(self.mlp_list[i]):
                layer_for_cur_radius['conv{}'.format(j)] = tf.keras.layers.Conv2D(num_out_channel,
                                                        kernel_size=[1, 1],
                                                        strides=(1, 1), padding='valid',
                                                        data_format=None, activation='relu',
                                                        use_bias=True,
                                                        kernel_regularizer=None)
            self.conv_layers['radius{}'.format(i)] = layer_for_cur_radius

    def call(self,input, training=True):
        xyz = input[0]
        points = input[1]
        new_xyz = gather_point(xyz, farthest_point_sample(self.npoint, xyz))
        new_points_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])

            if points is not None:
                grouped_points = group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            if self.use_nchw:
                grouped_points = tf.transpose(grouped_points, [0, 3, 1, 2])

            radius_id = 'radius{}'.format(i)
            for j, num_out_channel in enumerate(self.mlp_list[i]):
                conv_id = 'conv{}'.format(j)
                grouped_points = self.conv_layers[radius_id][conv_id](grouped_points)

            if self.use_nchw:
                grouped_points = tf.transpose(grouped_points, [0, 2, 3, 1])

            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)

        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


class PointNetPlusSetAbstaction(tf.keras.Model):
    ''' PointNet Set Abstraction (SA) Module
            Input:
                xyz: (batch_size, ndataset, 3) TF tensor
                points: (batch_size, ndataset, channel) TF tensor
                npoint: int32 -- #points sampled in farthest point sampling
                radius: float32 -- search radius in local region
                nsample: int32 -- how many points in each local region
                mlp: list of int32 -- output size for MLP on each point
                mlp2: list of int32 -- output size for MLP on each region
                group_all: bool -- group all points into one PC if set true, OVERRIDE
                    npoint, radius and nsample settings
                use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
                use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
            Return:
                new_xyz: (batch_size, npoint, 3) TF tensor
                new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
                idx: (batch_size, npoint, nsample) int32 -- indices for local regions
        '''
    def __init__(self, npoint, radius,
                 nsample, mlp, mlp2=None, group_all=False,
                 pooling='max', knn=False,
                 use_xyz=True, use_nchw=False):

        super(PointNetPlusSetAbstaction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw
        self.conv_layers = {}
        self.knn = knn
        self.pooling = pooling
        data_format = 'NCHW' if use_nchw else 'NHWC'

        for i, num_out_channel in enumerate(mlp):
            self.conv_layers['conv{}'.format(i)] = tf.keras.layers.Conv2D(num_out_channel,
                                                                          kernel_size=[1, 1],
                                                                          strides=(1, 1), padding='valid',
                                                                          data_format=None, activation='relu',
                                                                          use_bias=True,
                                                                          kernel_regularizer= None )#tf.keras.regularizers.l2(0.001))
    def call(self, input, training=True):
        xyz, points = input[0], input[1]
        if self.group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.knn, self.use_xyz)

            # Point Feature Embedding
        if self.use_nchw:
            new_points = tf.transpose(new_points, [0, 3, 1, 2])

        for i, num_out_channel in enumerate(self.mlp):
            new_points = self.conv_layers['conv{}'.format(i)](new_points)

        if self.use_nchw:
            new_points = tf.transpose(new_points, [0, 2, 3, 1])

        # Pooling in Local Regions
        if self.pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif self.pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif self.pooling == 'weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(exp_dists, axis=2,
                                                    keep_dims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif self.pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


class PointNetPlusEncoder(tf.keras.Model):
    def __init__(self, params,):
        super(PointNetPlusEncoder, self).__init__()
        self.params = params
        self.all_layer = {}
        self.fc_layer_seq = ['dn1', 'bn1', 'dp1', 'dn2', 'bn2', 'dp2', 'dn3']
        sa_layer = {}
        sa_layer['sa1'] = PointNetPlusSetAbstaction(npoint=512,
                                                         radius=0.2,
                                                         nsample=32,
                                                         mlp =[4*64, 4*64, 4*128],
                                                         use_nchw=True)
        sa_layer['sa2'] = PointNetPlusSetAbstaction(npoint=128,
                                                         radius=0.4,
                                                         nsample= 64,
                                                         mlp=[2*128, 2*128, 2*256],
                                                         use_nchw=False) #use_nchw=False
        sa_layer['sa3'] = PointNetPlusSetAbstaction( npoint=None, radius=None, nsample=None,
                                                         mlp=[256, 512, self.params['bneck_size']],
                                                         mlp2=None, use_nchw=False, ##use_nchw=False
                                                         group_all=True,)

        self.all_layer['sa'] = sa_layer
        # Fully connected layers
        fc_layer = {}
        fc_layer[self.fc_layer_seq[0]] = tf.keras.layers.Dense(512, activation='relu')
        fc_layer[self.fc_layer_seq[1]] = tf.keras.layers.BatchNormalization()
        fc_layer[self.fc_layer_seq[2]] = tf.keras.layers.Dropout(rate=0.5)
        fc_layer[self.fc_layer_seq[3]] = tf.keras.layers.Dense(256, activation='relu')
        fc_layer[self.fc_layer_seq[4]] = tf.keras.layers.BatchNormalization()
        fc_layer[self.fc_layer_seq[5]] = tf.keras.layers.Dropout(rate=0.5)
        fc_layer[self.fc_layer_seq[6]] = tf.keras.layers.Dense(40)
        self.all_layer['fc'] = fc_layer

    def call(self, input, is_training=True):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        if len(input)==1:
            l0_xyz = input[0]
            l0_points = None
        else:  l0_xyz, l0_points = input

        batch_size = tf.shape(l0_xyz)[0]#input.get_shape()[0].value
        num_point = l0_xyz.get_shape()[1].value
        end_points = {}

        # Set abstraction layers
        l1_xyz, l1_points, l1_indices = self.all_layer['sa']['sa1']([l0_xyz, l0_points])
        l2_xyz, l2_points, l2_indices = self.all_layer['sa']['sa2']([l1_xyz, l1_points])
        l3_xyz, l3_points, l3_indices = self.all_layer['sa']['sa3']([l2_xyz, l2_points])

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, self.params['bneck_size']])
        # for layer_id in self.fc_layer_seq:
        #     net = self.all_layer['fc'][layer_id](net)

        return net #, end_points

class PointNetPlusEncoderNoGlobal(tf.keras.Model):
    def __init__(self, params,):
        super(PointNetPlusEncoderNoGlobal, self).__init__()
        self.params = params
        self.all_layer = {}
        self.fc_layer_seq = ['dn1', 'bn1', 'dp1', 'dn2', 'bn2', 'dp2', 'dn3']
        sa_layer = {}
        sa_layer['sa1'] = PointNetPlusSetAbstaction(npoint=512,
                                                    radius=0.2,
                                                    nsample=32,
                                                    mlp =[64, 64, 128],
                                                    use_nchw=True)
        sa_layer['sa2'] = PointNetPlusSetAbstaction(npoint=256,
                                                    radius=0.4,
                                                    nsample= 64,
                                                    mlp=[128, 128, 3],
                                                    use_nchw=False) #use_nchw=False
        self.all_layer['sa'] = sa_layer

    def call(self, input, is_training=True):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = tf.shape(input)[0]#input.get_shape()[0].value
        num_point = input.get_shape()[1].value
        end_points = {}

        l0_xyz = input
        l0_points = None
        # Set abstraction layers
        l1_xyz, l1_points, l1_indices = self.all_layer['sa']['sa1']([l0_xyz, l0_points])
        l2_xyz, l2_points, l2_indices = self.all_layer['sa']['sa2']([l1_xyz, l1_points])

        # Fully connected layers
        net = tf.reshape(l2_xyz, [batch_size, 256*3])
        # for layer_id in self.fc_layer_seq:
        #     net = self.all_layer['fc'][layer_id](net)

        return net #, end_points




class PointNetPlusEncoderMSG(tf.keras.Model):
    def __init__(self, params,):
        super(PointNetPlusEncoderMSG, self).__init__()
        self.params = params
        self.all_layer = {}
        self.fc_layer_seq = ['dn1', 'bn1', 'dp1', 'dn2', 'bn2', 'dp2', 'dn3']
        sa_layer = {}
        sa_layer['sa1'] = PointNetPlusSetAbstactionMSG(npoint=512,
                                                         radius_list =[0.1, 0.2, 0.4],
                                                         nsample_list= [16, 32, 128],
                                                         mlp_list =[[32, 32, 64],[64, 64, 128],[64, 96, 128]],
                                                         is_training = True,
                                                         use_nchw=True)
        sa_layer['sa2'] = PointNetPlusSetAbstactionMSG(npoint=128,
                                                         radius_list =[0.2, 0.4, 0.8],
                                                         nsample_list= [32, 64, 128],
                                                         mlp_list =[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                         is_training = True,
                                                         use_nchw=False)
        sa_layer['sa3'] = PointNetPlusSetAbstaction( npoint=None, radius=None, nsample=None,
                                                         mlp=[256, 512, self.params['bneck_size']], #[256, 512, 1024]
                                                         mlp2=None, group_all=True,)

        self.all_layer['sa'] = sa_layer

        # Fully connected layers
        fc_layer = {}
        fc_layer[self.fc_layer_seq[0]] = tf.keras.layers.Dense(512, activation='relu')
        fc_layer[self.fc_layer_seq[1]] = tf.keras.layers.BatchNormalization()
        fc_layer[self.fc_layer_seq[2]] = tf.keras.layers.Dropout(rate=0.6)
        fc_layer[self.fc_layer_seq[3]] = tf.keras.layers.Dense(256, activation='relu')
        fc_layer[self.fc_layer_seq[4]] = tf.keras.layers.BatchNormalization()
        fc_layer[self.fc_layer_seq[5]] = tf.keras.layers.Dropout(rate=0.6)
        fc_layer[self.fc_layer_seq[6]] = tf.keras.layers.Dense(40)
        self.all_layer['fc'] = fc_layer



    def call(self, input, is_training=True):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = tf.shape(input)[0]  # input.get_shape()[0].value
        num_point = input.get_shape()[1].value
        end_points = {}

        l0_xyz = input
        l0_points = None
        # Set abstraction layers
        l1_xyz, l1_points = self.all_layer['sa']['sa1']([l0_xyz, l0_points])
        l2_xyz, l2_points = self.all_layer['sa']['sa2']([l1_xyz, l1_points])
        l3_xyz, l3_points, _ = self.all_layer['sa']['sa3']([l2_xyz, l2_points])

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, self.params['bneck_size']])
        # for layer_id in self.fc_layer_seq[:2]:
        #     net = self.all_layer['fc'][layer_id](net)

        return net#, end_points

