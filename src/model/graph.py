import tensorflow as tf
import tensorflow.experimental.numpy as tnp

import dgl
from dgl import DGLGraph
from functools import partial
from helper.metric import tf_batchdot

from config import MODEL

class Graph(tf.keras.layers.Layer):
    """ The base class of GAT and MetaGAT. We implement the methods based on DGL library. """

    @staticmethod
    def create(graph_type, dist, edge, hidden_size, name):
        """ create a graph. """
        if graph_type == 'None': return None
        elif graph_type == 'GAT': return GAT(dist, edge, hidden_size, name=name)
        elif graph_type == 'MetaGAT': return MetaGAT(dist, edge, hidden_size, name=name)
        else: raise Exception('Unknow graph: %s' % graph_type)

    @staticmethod
    def create_graphs(graph_type, graph, hidden_size, name):
        """ Create a list of graphs according to graph_type & graph. """
        if graph_type == 'None': return None
        dist, e_bi = graph
        return [
            #Graph.create(graph_type, dist.T, e_bi, hidden_size, name + 'in_'),
            Graph.create(graph_type, dist, e_bi, hidden_size, name + 'out_')
        ]

    def __init__(self, dist, edge, hidden_size, name=None):
        super(Graph, self).__init__(name=name)
        self.dist = dist #correlation_matrix
        self.edge = edge #e_bi - connected node ids for each node
        self.hidden_size = hidden_size

        # create graph
        self.num_nodes = n = 50 #self.dist.shape[0]
        src, dst, dist = [], [], []
        for i in range(n): # for each node
            for j in edge[i]: # for each edge for node i (j = dst node id)
                src.append(j) # from row #
                dst.append(i) # to coloumn #
                dist.append(self.dist[j, i]) # value of [row, column] in correlation matrix

        self.src = src
        self.dst = dst
        self.dist = tf.expand_dims(tf.convert_to_tensor(dist), axis=1)
        self.ctx = []
        self.graph_on_ctx = []

        self.init_model()

    def build_graph_on_ctx(self, ctx):
        g = DGLGraph()
        g.set_n_initializer(dgl.init.zero_initializer)
        g.add_nodes(self.num_nodes)
        g.add_edges(self.src, self.dst)
        with tf.device(ctx):
            g.edata['dist'] = self.dist # assigns attribute dist[i] to edges [src[i],dst[i]]
        self.graph_on_ctx.append(g)
        self.ctx.append(ctx)



    def get_graph_on_ctx(self, ctx):
        if ctx not in self.ctx:
            self.build_graph_on_ctx(ctx)
        return self.graph_on_ctx[self.ctx.index(ctx)]

    def call(self, state, feature): # first dimension of state & feature should be num_nodes
        g = self.get_graph_on_ctx('GPU:0')
        #print('graph state shape',state.shape)
        #print('graph feature shape', feature.shape)
        g.ndata['state'] = state    # assigns node features for all nodes
        g.ndata['feature'] = feature
        g.update_all(self.msg_edge, self.msg_reduce) # send messages along all the edges and update all the nodes
        state = g.ndata.pop('new_state')
        return state

    def init_model(self):
        raise NotImplementedError("To be implemented")

    def msg_edge(self, edge):
        """ Messege passing across edge.
        More detail usage please refers to the manual of DGL library.

        Parameters
        ----------
        edge: a dictionary of edge data.
            edge.src['state'] and edge.dst['state']: hidden states of the nodes, which is NDArrays with shape [e, b, t, d] or [e, b, d]
            edge.src['feature'] and  edge.dst['state']: features of the nodes, which is NDArrays with shape [e, d]
            edge.data['dist']: distance matrix of the edges, which is a NDArray with shape [e, d]

        Returns
        -------
            A dictionray of messages
        """
        raise NotImplementedError("To be implemented")

    def msg_reduce(self, node):
        raise NotImplementedError("To be implemented")

class GAT(Graph):
    def __init__(self, dist, edge, hidden_size, name=None):
        super(GAT, self).__init__(dist, edge, hidden_size, name)

    def init_model(self):
        self.weight = tf.Variable(tf.zeros([self.hidden_size * 2, self.hidden_size]), name='weight', shape=(self.hidden_size * 2, self.hidden_size))

    def msg_edge(self, edge):
        state = tnp.concatenate((edge.src['state'], edge.dst['state']), -1)
        ctx = 'GPU:0'

        alpha = tf.nn.leaky_relu(tf.tensordot(state, self.weight, axes=1))

        dist = edge.data['dist']
        while len(dist.shape) < len(alpha.shape):
            dist = tnp.expand_dims(dist, axis=-1)

        alpha = alpha * dist
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = tf.nn.softmax(alpha, axis=1)

        new_state = tf.nn.relu(tf.math.reduce_sum(alpha * state, axis=1))
        return { 'new_state': new_state }

class MetaGAT(Graph):
    """ Meta Graph Attention. """
    def __init__(self, dist, edge, hidden_size, name=None):
        super(MetaGAT, self).__init__(dist, edge, hidden_size, name)

    def init_model(self):
        from model.basic_structure import MLP
        self.w_mlp = MLP(MODEL['meta_hiddens'] + [self.hidden_size * self.hidden_size * 2,], 'sigmoid', False)
        self.weight = tf.Variable(tf.zeros([1,1]), name='weight', shape=(1,1))

    def msg_edge(self, edge):
        state = tf.concat([edge.src['state'], edge.dst['state']], -1)
        #print('edge.src state shape', edge.src['state'].shape)
        #print('edge.dst state shape', edge.dst['state'].shape)
        #print('edge.src feature shape', edge.src['feature'].shape)
        #print('edge dst feature shape', edge.dst['feature'].shape)
        feature = tf.concat([edge.src['feature'], edge.dst['feature'], tf.cast(edge.data['dist'], tf.float32)], -1)

        # generate weight by meta-learner
        weight = self.w_mlp(feature)
        #print('weight shape', weight.shape)
        weight = tf.reshape(weight, shape=(-1, self.hidden_size * 2, self.hidden_size))
        #print('weight reshape', weight.shape)

        # reshape state to [n, b * t, d] for batch_dot (currently mxnet only support batch_dot for 3D tensor)
        shape = state.shape
        #print('state shape', shape)
        state = tf.reshape(state, shape=(shape[0], -1, shape[-1])) # []
        #print('state reshape', state.shape)
        alpha = tf.nn.leaky_relu(tf_batchdot(state, weight))
        #print('alpha', alpha.shape)
        # reshape alpha to [n, b, t, d]
        alpha = tf.reshape(alpha, shape=shape[:-1] + (self.hidden_size,))
        return { 'alpha': alpha, 'state': edge.src['state'] }

    def msg_reduce(self, node):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = tf.nn.softmax(alpha, axis=1)
        #print('MetaGAT weight', self.weight)
        new_state = tf.nn.relu(tf.reduce_sum(alpha * state, axis=1)) * tf.math.sigmoid(self.weight)
        return { 'new_state': new_state }
