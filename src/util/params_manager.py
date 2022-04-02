#create a param dict from reading a yml file
import numpy as np
INPUT_SHAPE = [2048, 3]
FC_DECODER_OUT_SHAPE = [1024, 3] #[2048, 3] #
BNECK_SIZE = 512
encoder_params = {'n_filters': [2*64, 2*128, 2*128, 2*256, BNECK_SIZE], #[64, 128, 128, 256, BNECK_SIZE]#[32, 64, 128, 2],#
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'padding': 'same',
                    'symmetry' : True,
                    'bneck_size': BNECK_SIZE
                  }

decoder_params = {  'layer_sizes': [512, 512, 1024, np.prod(FC_DECODER_OUT_SHAPE)],#[256, 256, np.prod(input_shape)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'out_shape': FC_DECODER_OUT_SHAPE
                 }
train_params = {    'batch_size': 16, #50
                    'mx_epochs': 1000,
                    'learning_rate': 2e-4, #.0005
                    'z_rotate': True,
                    'gauss_augment': False,
                    'saver_step': 100,#10
                    'loss_display_step': 1,
                    'loss': 'chamfer', #'emd' ,#
                    'val_split': 0.1,
                    'scale_loss':False,
                  }

params={}
params['encoder_params'] = encoder_params
params['decoder_params'] = decoder_params
params['train_params'] = train_params
