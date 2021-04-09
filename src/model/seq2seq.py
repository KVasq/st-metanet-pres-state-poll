import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


from model.cell import RNNCell
from model.graph import Graph
from model.basic_structure import MLP
from helper.metric import tf_mean_exclude

class Encoder(tf.keras.layers.Layer):
    """ Seq2Seq encoder. """
    def __init__(self, cells, graphs, name=None):
        super(Encoder, self).__init__(name=name)

        self.cells = cells
        """for cell in cells:
            self.register_child(cell)"""

        self.graphs = graphs
        """
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)"""

    def call(self, feature, data):
        """ Encode the temporal sequence sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, t, d].

        n = nth item (nth node) (n,d) 1D vector at each depth
        d = hidden size / num of features
        t = timestep (n,b,t,d) 3D vector at each depth
        b = batch_size

        Returns
        -------
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        """

        _, batch, seq_len, _ = data.shape
        states = []
        for depth, cell in enumerate(self.cells):
            # rnn unroll
            #print('encoder data input', tf.shape(data))
            data, state = cell(feature, data, None) # whole_sequence_output [batch_size, timesteps, units], final_state [batch_size, num_hidden]
            states.append(state)

            # graph attention
            if self.graphs[depth] != None:
                _data = 0
                #print('encoder graph call')
                for g in self.graphs[depth]:
                    _data = _data + g(data, feature)
                data = _data

        return states

class Decoder(tf.keras.layers.Layer):
    """ Seq2Seq decoder. """
    def __init__(self, cells, graphs, input_dim, output_dim, use_sampling, cl_decay_steps, name=None):
        super(Decoder, self).__init__(name=name)
        self.cells = cells
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_sampling = use_sampling
        self.global_steps = 0.0
        self.cl_decay_steps = float(cl_decay_steps)

        """for cell in cells:
            self.register_child(cell)"""

        self.graphs = graphs
        """
        for graph in graphs:
            if graph is not None:
                for g in graph:
                    if g is not None:
                        self.register_child(g)"""

        # initialize projection layer for the output
        with tf.name_scope('decoder'):
            self.proj = tf.keras.layers.Dense(output_dim, name='proj_')

    def sampling(self):
        """ Schedule sampling: sampling the ground truth. """
        threshold = self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_steps / self.cl_decay_steps))
        return float(random.random() < threshold)

    def call(self, feature, label, begin_states, is_training):
        ''' Decode the hidden states to a temporal sequence.

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        label: a NDArray with shape [n, b, t, d].
        begin_states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        is_training: bool

        Returns
        -------
            outputs: the prediction, which is a NDArray with shape [n, b, t, d]
        '''
        ctx = label.device[-5:]

        num_nodes, batch_size, seq_len, _ = label.shape
        aux = label[:,:,:, self.output_dim:] # [n,b,t,d]
        label = label[:,:,:, :self.output_dim] # [n,b,t,d]

        go = tf.zeros(shape=(num_nodes, batch_size, self.input_dim))
        states = begin_states
        output = tnp.array([], dtype= tf.float64)
        for i in range(seq_len):
            # get next input
            if i == 0: data = go
            else:
                #output = tf.cast(output, tf.float64)
                #aux = tf.cast(aux, tf.float64)
                #print('decoder output', output[i - 1])
                #print('decoder aux', aux[:,:,i - 1][i][i])

                prev = tf.concat([output[i - 1], aux[:,:,i - 1]], axis=-1)
                #print('decoder prev', tf.shape(prev))
                truth = tf.concat([label[:,:,i - 1], aux[:,:,i - 1]], axis=-1)
                if is_training and self.use_sampling: value = self.sampling()
                else: value = 0
                data = value * truth + (1 - value) * prev

            # unroll 1 step
            for depth, cell in enumerate(self.cells):
                #print('decoder forward_single data input', tf.shape(data))

                if tf.shape(states[depth])[0] == 32:
                    states[depth] = tf.transpose(states[depth], (3, 1, 2, 0))
                #print('decoder forward_single state input', tf.shape(states[depth])) # transpose this states??
                data, states[depth] = cell.forward_single(feature, data, states[depth])
                if self.graphs[depth] is not None:
                    _data = 0
                    #print('decoder graph call')
                    for g in self.graphs[depth]:
                        _data = _data + g(data, feature)
                    data = _data / len(self.graphs[depth])

            # append feature to output
            #print('decoder feature', tf.shape(feature))
            _feature = tf.expand_dims(feature, axis=1) # [n, 1, d]
            #print('decoder feature expanded', tf.shape(_feature))
            hidden_num = tf.shape(_feature)[2]
            _feature = tf.broadcast_to(_feature, shape=(num_nodes, batch_size, hidden_num)) # [n, b, d]
            data = tf.concat([data, _feature], axis=-1) # [n, b, t, d]

            # proj output to prediction
            data = tf.reshape(data, shape=(num_nodes * batch_size, -1))
            data = self.proj(data)
            data = tf.reshape(data, shape=(num_nodes, batch_size, -1))
            #output = output.numpy().tolist()
            #print('data output', tf.shape(data))
            if len(tf.shape(output)) < 3:
                output = tf.reshape(output, (0, 51, tf.shape(data)[1], 2))
            output = tf.concat([output, [data]], axis=0)
            #output.append(data)
            #print('decoder output loop end', tf.shape(output))


        output = tf.stack([*output], axis=2)
        return output

class Seq2Seq(tf.keras.Model):
    def __init__(self,
                 geo_hiddens,
                 rnn_type, rnn_hiddens,
                 graph_type, graph,
                 input_dim, output_dim,
                 use_sampling,
                 cl_decay_steps,
                 name=None):
        """ Initializer.

        Parameters
        ----------
        geo_hiddens: list of int
            the hidden units of NMK-learner.
        rnn_type: list of str
            the types of rnn cells (denoting GRU or MetaGRU).
        rnn_hiddens: list of int
            the hidden units for each rnn layer.
        graph_type: list of str
            the types of graph attention (denoting GAT or MetaGAT).
        graph: the graph structure
        input_dim: int
        output_dim: int
        use_sampling: bool
            whether use schedule sampling during training process.
        cl_decay_steps: int
            decay steps in schedule sampling.
        """
        super(Seq2Seq, self).__init__(name=name)

        # initialize encoder
        with tf.name_scope('encoder'):
            encoder_cells = []
            encoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, name='encoder_c%d_' % i)
                g = Graph.create_graphs('None' if i == len(rnn_hiddens) - 1 else graph_type[i], graph, hidden_size, name='encoder_g%d_' % i)
                encoder_cells.append(c)
                encoder_graphs.append(g)
        self.encoder = Encoder(encoder_cells, encoder_graphs)

        # initialize decoder
        with tf.name_scope('decoder'):
            decoder_cells = []
            decoder_graphs = []
            for i, hidden_size in enumerate(rnn_hiddens):
                pre_hidden_size = input_dim if i == 0 else rnn_hiddens[i - 1]
                c = RNNCell.create(rnn_type[i], pre_hidden_size, hidden_size, name='decoder_c%d_' % i)
                g = Graph.create_graphs(graph_type[i], graph, hidden_size, name='decoder_g%d_' % i)
                decoder_cells.append(c)
                decoder_graphs.append(g)
        self.decoder = Decoder(decoder_cells, decoder_graphs, input_dim, output_dim, use_sampling, cl_decay_steps)

        # initalize geo encoder network (node meta knowledge learner)
        self.geo_encoder = MLP(geo_hiddens, act_type='relu', out_act=True, name='geo_encoder_')

    def meta_knowledge(self, feature):
        return self.geo_encoder(tf.reduce_mean(feature, axis=0))

    def call(self, feature, data, label, mask, is_training):
        """ Forward the seq2seq network.

        Parameters
        ----------
        feature: NDArray with shape [b, n, d].
            The features of each node.
        data: NDArray with shape [b, t, n, d].
            The flow readings.
        label: NDArray with shape [b, t, n, d].
            The flow labels.
        is_training: bool.


        Returns
        -------
        loss: loss for gradient descent.
        (pred, label): each of them is a NDArray with shape [n, b, t, d].

        """

        #print('data pre-transposed', data.shape)
        data = tf.transpose(data, (2, 0, 1, 3)) # [n, b, t, d] node, batch size, timesteps, features
        label = tf.transpose(label, (2, 0, 1, 3)) # [n, b, t, d]
        mask = tf.transpose(mask, (2, 0, 1, 3)) # [n, b, t, d]
        #print('data transposed', tf.shape(data))
        # geo-feature embedding (NMK Learner)
        #print('feature pre encoded',feature)

        feature = self.geo_encoder(np.mean(feature, axis=0)) # shape=[n, d]
        #print('feature encoded', feature)

        # seq2seq encoding process
        states = self.encoder(feature, data)
        #for i in states:
            #print('encoder state return', tf.shape(i)) # [32 51 4 1]
        # seq2seq decoding process
        #print('decoder states input', tf.shape(states))  # [2 32 51 4 1] (1 state per cell)
        #print('decoder label input', tf.shape(label)) # [51 4 12 2]
        #print('training bool', is_training)
        output = self.decoder(feature, label, states, is_training) # [n, b, t, d]

        # loss calculation
        label = label[:,:,:,:self.decoder.output_dim]
        mask = tf.cast(mask, tf.float64)
        #print('seq2seq output', output)
        #print('label', label)
        #print('seq2seq mask', mask)
        output = output * mask
        label = label * mask

        loss = tf_mean_exclude(tf.abs(output - label), axis=1) # exclude batch axis
        #print('loss', loss)
        return loss, [output, label, mask] #output is prediction

def net(settings):
    from data.dataloader import get_soc_feature
    _, graph = get_soc_feature(settings['dataset'])

    net = Seq2Seq(
        geo_hiddens = settings['model']['geo_hiddens'],
        rnn_type    = settings['model']['rnn_type'],
        rnn_hiddens = settings['model']['rnn_hiddens'],
        graph_type  = settings['model']['graph_type'],
        graph       = graph,
        input_dim   = settings['dataset']['input_dim'],
        output_dim  = settings['dataset']['output_dim'],
        use_sampling    = settings['training']['use_sampling'],
        cl_decay_steps  = settings['training']['cl_decay_steps'],
        name     = settings['model']['type'] + "_"
    )
    return net
