import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from model.basic_structure import MetaDense
from config import MODEL

class RNNCell(tf.keras.layers.Layer):
    def __init__(self, name):
        super(RNNCell, self).__init__(name=name)

    @staticmethod
    def create(rnn_type, pre_hidden_size, hidden_size, name):
        if rnn_type == 'MyGRUCell': return MyGRUCell(hidden_size, name)
        elif rnn_type == 'MetaGRUCell': return MetaGRUCell(pre_hidden_size, hidden_size, meta_hiddens=MODEL['meta_hiddens'], name=name)
        else: raise Exception('Unknown rnn type: %s' % rnn_type)

    def forward_single(self, feature, data, begin_state):
        """ Unroll the recurrent cell with one step

        Parameters
        ----------
        data: a NDArray with shape [n, b, d].
        feature: a NDArray with shape [n, d].
        begin_state: a NDArray with shape [n, b, d]

        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

        """
        raise NotImplementedError("To be implemented")

    def call(self, feature, data, begin_state):
        """ Unroll the temporal sequence sequence.

        Parameters
        ----------
        data: a NDArray with shape [n, b, t, d].
        feature: a NDArray with shape [n, d].
        begin_state: a NDArray with shape [n, b, d]

        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, t, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

        """
        raise NotImplementedError("To be implemented")


class MyGRUCell(RNNCell):
    """ A common GRU Cell. """
    def __init__(self, hidden_size, name=None):
        super(MyGRUCell, self).__init__(name=name)
        self.hidden_size = hidden_size
        with tf.name_scope('GRU'):
            self.cell = tf.keras.layers.GRU((self.hidden_size), return_sequences=True, return_state=True, name='GRU')

    def forward_single(self, feature, data, begin_state):
        # add a temporal axis
        data = tf.expand_dims(data, axis=2)

        # unroll
        data, state = self(feature, data, begin_state)

        # remove the temporal axis
        data = tf.reduce_mean(data, axis=2)

        return data, state

    def call(self, feature, data, begin_state):
        n, b, length, _ = data.shape

        # reshape the data and states for rnn unroll
        data = tf.reshape(data, shape=(n * b, length, -1)) # [n * b, t, d]
        if begin_state is not None:
            begin_state = [
                tf.reshape(state, shape=(n * b, -1)) for state in begin_state
            ] # [n * b, d]
            begin_state = tf.transpose(begin_state, (1,0,2))
            begin_state = tf.squeeze(begin_state)

        # unroll the rnn
        #print('GRU: data unroll shape', data.shape)
        #print('GRU: unrolling rnn')
        #print('begin state', begin_state)
        #print('GRU: begin state shape', np.array(begin_state).shape)
        #print('GRU: get initial state', self.cell.get_initial_state(data))

        data, state = self.cell(data, initial_state=begin_state)
        #print('GRUCell pass %d:' % _, 'data', data.shape, 'state', state.shape)
        #print('GRU: unrolling finished')
        #print(data.shape)
        # reshape the data & states back
        data = tf.reshape(data, shape=(n, b, length, -1))
        #print('GRU: cell data return', data.shape)
        state = tf.transpose(state) # this is messing up states for decoder
        state = [tf.reshape(s, shape=(n, b, -1)) for s in state]
        #print('GRU: cell state return', np.array(state).shape)
        return data, state

class MetaGRUCell(RNNCell):
    """ Meta GRU Cell. """

    def __init__(self, pre_hidden_size, hidden_size, meta_hiddens, name=None):
        super(MetaGRUCell, self).__init__(name=name)
        self.pre_hidden_size = pre_hidden_size
        self.hidden_size = hidden_size
        with tf.name_scope('MGRU'):
            self.dense_z = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, name='dense_z_')
            self.dense_r = MetaDense(pre_hidden_size + hidden_size, hidden_size, meta_hiddens=meta_hiddens, name='dense_r_')

            self.dense_i2h = MetaDense(pre_hidden_size, hidden_size, meta_hiddens=meta_hiddens, name='dense_i2h_')
            self.dense_h2h = MetaDense(hidden_size, hidden_size, meta_hiddens=meta_hiddens, name='dense_h2h_')

    def forward_single(self, feature, data, begin_state):
        """ unroll one step

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, d].
        begin_state: a NDArray with shape [n, b, d].

        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.

        """
        if begin_state is None:
            num_nodes, batch_size, _ = data.shape
            begin_state = [tnp.zeros((num_nodes, batch_size, self.hidden_size))]

        prev_state = begin_state[0]
        #print('MGRU forward_single: previous state', prev_state.shape)
        #print('MGRU forward_single: data', tf.shape(data))
        data_and_state = tf.concat([data, prev_state], axis=-1)
        #print('MGRU forward_single: data and state metadense input', tf.shape(data_and_state))
        z = tf.math.sigmoid(self.dense_z(feature, data_and_state))
        r = tf.math.sigmoid(self.dense_r(feature, data_and_state))

        state = z * prev_state + (1 - z) * tf.tanh(self.dense_i2h(feature, data) + self.dense_h2h(feature, r * prev_state))

        return state, [state]

    def call(self, feature, data, begin_state):
        num_nodes, batch_size, length, _ = data.shape

        data = tf.split(data, axis=2, num_or_size_splits=length)
        #print('data split', tf.shape(data))
        data = tf.squeeze(data, axis=3)

        #print('MGRU: data split and squeeze', tf.shape(data))
        outputs, state = [], begin_state
        for input in data: # for each timestep
            output, state = self.forward_single(feature, input, state)
            outputs.append(output)
            #print('MGRU: output appended')

        outputs = tf.stack([*outputs], axis=2)
        #print('MetaGru outputs/data return', tf.shape(outputs))
        #print('MetaGru state return', tf.shape(state))
        state = tf.transpose(state, (3,1,2,0)) #
        return outputs, state
