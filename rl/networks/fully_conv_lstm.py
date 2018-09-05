import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn

from pysc2.lib import actions
from pysc2.lib import features

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES
from rl.networks.fully_conv import FullyConv


class FullyConvLSTM(FullyConv):
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def conv2dlstm(self, x, h):
    x = self.to_nhwc(x)  # NCHW conv not available for Conv2DLSTM

    # conv2dlstm operation
    layer = rnn.Conv2DLSTMCell(
        input_shape=[32, 32, 75],
        output_channels=96,
        kernel_shape=[3, 3],)
    y, h = layer.apply(x, h)
    y = tf.nn.relu(y)
    y = self.from_nhwc(y)

    return y, h

  def build(self, screen_input, minimap_input, flat_input, state):
    size2d = tf.unstack(tf.shape(screen_input)[1:3])
    screen_emb = self.embed_obs(screen_input, features.SCREEN_FEATURES,
                                self.embed_spatial)
    minimap_emb = self.embed_obs(minimap_input, features.MINIMAP_FEATURES,
                                 self.embed_spatial)
    flat_emb = self.embed_obs(flat_input, FLAT_FEATURES, self.embed_flat)

    screen_out = self.input_conv(self.from_nhwc(screen_emb), 'screen')
    minimap_out = self.input_conv(self.from_nhwc(minimap_emb), 'minimap')

    broadcast_out = self.broadcast_along_channels(flat_emb, size2d)

    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])
    # we add an LSTM on state_out. state is previous cell and hidden state
    state_out, new_state = self.conv2dlstm(state_out, state)

    flat_out = layers.flatten(self.to_nhwc(state_out))
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)

    value = layers.fully_connected(fc, 1, activation_fn=None)
    value = tf.reshape(value, [-1])

    fn_out = self.non_spatial_output(fc, NUM_FUNCTIONS)

    args_out = dict()
    for arg_type in actions.TYPES:
      if is_spatial_action[arg_type]:
        arg_out = self.spatial_output(state_out)
      else:
        arg_out = self.non_spatial_output(fc, arg_type.sizes[0])
      args_out[arg_type] = arg_out

    policy = (fn_out, args_out)

    return policy, value, new_state
