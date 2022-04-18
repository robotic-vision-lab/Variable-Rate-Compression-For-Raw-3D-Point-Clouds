import tensorflow as tf
import os,sys
import numpy as np
import tensorflow_compression as tfc

from external.structural_losses.tf_nndistance import nn_distance
from external.structural_losses.tf_approxmatch import approx_match, match_cost
from . import decoders
from . import encoders


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))


class AutoEncoder(tf.keras.Model):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.lamda = 1e-5 #0.0001
        self.encoder = encoders.PointNetPlusEncoder(params['encoder_params']) #PointNetEncoder(params['encoder_params'])
        self.entropy_bottleneck = tfc.EntropyBottleneck()#filters=(128, 128, 16)
        self.decoder = decoders.FC_Decoder(params['decoder_params']) #FC_Decoder#decoders.FoldingNetDecoder()# #decoders.SharedFCDecoder()  #
        self.aux_encoder = encoders.PointNetEncoder(params['encoder_params'])
        self.aux_decoder = decoders.UpConvDecoder()
        self.embedding_sharing1 = tf.keras.layers.Dense(2*params['encoder_params']['bneck_size'])
        self.embedding_sharing2 = tf.keras.layers.Dense(2*params['encoder_params']['bneck_size'])

        # # self.embedding_sharing = autoregressive_dense.MaskingDense(1024, 1024,
        # #                                       hidden_layers=1,
        # #                                       random_input_order=False)
        #
        # self.autoregressive_decoder = autoregressive_dense.MaskingDense(2048*3, 2048*3,
        #                                       hidden_layers=3,
        #                                       random_input_order=False)

    def feature_embedding(self, x):
        freq = 6
        freq_bands = tf.tile(2.**tf.linspace(0., freq-1, freq),[3])
        x_prime = tf.repeat(x, repeats=[freq, freq, freq], axis=2 )#repeats=[freq, freq, freq]
        x_prime = tf.multiply(x_prime, freq_bands)
        embed = tf.concat([tf.sin(x_prime),tf.cos(x_prime)], axis=-1)

        return embed

    def get_truncated_embedding(self, embedding, embedding_len):
        embedding_size = embedding.get_shape()[1]
        embedding = embedding[:, :embedding_len]
        embedding = tf.pad(embedding, [[0,0], [0, embedding_size- embedding_len ]], "CONSTANT")
        embedding = tf.reshape(embedding,[-1,1024])
        return embedding


    def call(self, inputs, training=True):
        enable_freq_embedding = True
        pc_xyz, label = inputs
        self.transformation = None
        # if experiment frequency embedding
        if enable_freq_embedding:
            x_embed = self.feature_embedding(pc_xyz)
            pn_plus_z = self.encoder([pc_xyz, x_embed])
            pn_z = self.aux_encoder(tf.concat([pc_xyz, x_embed], axis=-1))
        else:
            pn_plus_z = self.encoder([pc_xyz])
            pn_z = self.aux_encoder(pc_xyz)

        embedding = tf.concat((pn_plus_z, pn_z ), axis=-1)
        embedding_size = embedding.get_shape()[1]
        embedding = self.embedding_sharing1(embedding)

        # effective_embedding_size = tf.random.uniform([1], minval=128, maxval=embedding_size, dtype=tf.int32, seed=2, name=None)[0]
        # embedding = embedding[:, :effective_embedding_size]
        # embedding = tf.pad(embedding, [[0,0], [0, embedding_size- effective_embedding_size ]], "CONSTANT")
        # embedding = tf.reshape(embedding,[-1,1024])

        if training:
            embedding, likelihoods = self.entropy_bottleneck(embedding, training=True)
            self.likelihoods = likelihoods
        else: #compressing, decompressing
            if 'VEC_LENGTH' in globals():
                embedding = self.get_truncated_embedding(embedding, VEC_LENGTH)
            self.embedding_string = self.entropy_bottleneck.compress(embedding)
            embedding = self.entropy_bottleneck.decompress(self.embedding_string, [1024] )#tf.shape(embedding)[1:],#channels=self.num_filters
        #embedding = tf.concat((label, embedding ), axis=-1)

        self.embedding = embedding
        #embedding = embedding[:,512:]
        batch_size = embedding.get_shape()[0]
        embedding_size = embedding.get_shape()[1]
        # decoder_expected_size = 2048*3#1024#2048*3
        self.w = self.latent_code_weight(1024)
        # embedding = tf.pad(embedding, [[0,0], [0, decoder_expected_size- embedding_size ]], "CONSTANT")
        # embedding = tf.reshape(embedding,[-1, decoder_expected_size])
        x_tilde = self.decoder(embedding)#self.autoregressive_decoder(embedding)#self.decoder(embedding)
#        x_tilde = tf.reshape(x_tilde, [batch_size, -1, 3]) #16-- batch size
        x_tilde_upconv = self.aux_decoder(embedding) # pointnet plus output to upconv input
        x_tilde = tf.concat((x_tilde, x_tilde_upconv), axis=1)

        return x_tilde


    def decompress(self, embedding_string):
        embedding = self.entropy_bottleneck.decompress(embedding_string, [1024],channels=1024)
        x_tilde = self.decoder(embedding[:, :self.params['encoder_params']['bneck_size']])
        x_tilde_upconv = self.aux_decoder(embedding[:, self.params['encoder_params']['bneck_size']:]) # pointnet plus output to upconv input

        x_tilde = tf.concat((x_tilde, x_tilde_upconv), axis=1)

        return x_tilde

    def latent_code_weight(self, latent_code_length):

        w = np.arange(latent_code_length, dtype=np.float32)
        #w = (2e-4)*(w**1.8)+1.
        #w = 1.48-.00545*w+.0000228*w**2
        w =  15*np.exp(-.003*w)
        w = tf.constant(w, 'float32')
        return w

    def calc_loss(self, X, X_tilde):

        if self.params['train_params']['scale_loss']:
            scale = self.transformation[1]
            scale = tf.expand_dims(scale, axis=-1)
            X = tf.multiply(self.scale, X)
            X_tilde = tf.multiply(self.scale, X_tilde)

        if self.params['train_params']['loss'] == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(X_tilde, X)
            loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif self.params['train_params']['loss'] == 'emd':
            match = approx_match(X_tilde, X)
            loss = tf.reduce_mean(match_cost(X_tilde, X, match))
        w = tf.reshape(self.w, [1024,1]) #self.latent_code_weight(1024)
        weighted_entropy_loss = -tf.reduce_sum(tf.matmul(tf.log(self.likelihoods), w))
        weighted_entropy_loss = weighted_entropy_loss/(tf.log(2.) * tf.cast(tf.shape(X)[0]*tf.shape(X)[1], tf.float32)) # divide by number of points

        entropy_loss = -tf.reduce_sum(tf.log(self.likelihoods))
        entropy_loss = entropy_loss/(tf.log(2.) * tf.cast(tf.shape(X)[0]*tf.shape(X)[1], tf.float32)) # divide by number of points

        self.entropy_loss = entropy_loss
        #self.weighted_entropy_loss = weighted_entropy_loss
        self.chamfer_loss = 100*loss
        self.add_metric(loss, aggregation='mean', name='{}_loss'.format(self.params['train_params']['loss']))
        self.add_metric(entropy_loss, aggregation='mean', name='entropy_loss')
        return 100*loss + self.lamda * entropy_loss#weighted_entropy_loss
        #return 100 * loss + self.lamda *  weighted_entropy_loss

    def setup_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params['train_params']['learning_rate'])
        main_step = optimizer.minimize(loss)

        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(self.entropy_bottleneck.losses[0])

        train_step = tf.group(main_step, aux_step, self.entropy_bottleneck.updates[0])
        return train_step

