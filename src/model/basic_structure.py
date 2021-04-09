import tensorflow as tf
from helper.metric import tf_batchdot

class MLP(tf.keras.Sequential):
# Multilayer perceptron.
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        The initializer.

        Parameters
        ----------
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Weather apply activation function after last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        with tf.name_scope("MLP"):
            for i, h in enumerate(hiddens):
                activation = None if i == len(hiddens) - 1 and not out_act else act_type
                self.add(tf.keras.layers.Dense(h, activation=activation, kernel_initializer=weight_initializer))

class MetaDense(tf.keras.layers.Layer):
# The meta-dense layer. """
    def __init__(self, input_hidden_size, output_hidden_size, meta_hiddens, name=None):
        """
        The initializer.

        Parameters
        ----------
        input_hidden_size: int
            The hidden size of the input.
        output_hidden_size: int
            The hidden size of the output.
        meta_hiddens: list of int
            The list of hidden units of meta learner (a MLP).
        """
        super(MetaDense, self).__init__(name=name)
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.act_type = 'sigmoid'

        with tf.name_scope("MetaDense"):
            self.w_mlp = MLP(meta_hiddens + [self.input_hidden_size * self.output_hidden_size,], act_type=self.act_type, out_act=False, name='w_')
            self.b_mlp = MLP(meta_hiddens + [1,], act_type=self.act_type, out_act=False, name='b_')

    def __call__(self, feature, data):
        """ Forward process of a MetaDense layer

        Parameters
        ----------
        feature: NDArray with shape [n, d]
        data: NDArray with shape [n, b, input_hidden_size]

        Returns
        -------
        output: NDArray with shape [n, b, output_hidden_size]
        """
        #print('metadense feature', tf.shape(feature))
        weight = self.w_mlp(feature) # [n, input_hidden_size * output_hidden_size]
        weight = tf.reshape(weight, (-1, self.input_hidden_size, self.output_hidden_size))
        #print('metadense weight', tf.shape(weight))
        #print('metadense data', tf.shape(data))
        bias = tf.reshape(self.b_mlp(feature), shape=(-1, 1, 1)) # [n, 1, 1]
        #print('metadense return', tf.shape(tf_batchdot(data, weight) + bias))
        return tf.matmul(data, weight) + bias
