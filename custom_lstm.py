import tensorflow as tf
from tensorflow.keras.regularizers import l2

# Heavily inspired by Philippe Rémy's repo "cond_rnn":
# https://github.com/philipperemy/cond_rnn


def _get_tensor_shape(t):
    return t.shape


class CondLSTM(tf.keras.layers.Layer):

    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, *args, **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None

        self.rnn = tf.keras.layers.LSTM(self.units, *args, **kwargs)

        # single cond
        self.cond_to_init_state_dense_1 = tf.keras.layers.Dense(
            units=self.units)

        # multi cond
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for _ in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(
                tf.keras.layers.Dense(units=self.units))
        self.multi_cond_p = tf.keras.layers.Dense(
            1, activation=None, use_bias=True)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if first_cond_dim == 1:
            initial_cond = tf.tile(initial_cond, [2, 1, 1])
        elif first_cond_dim != 2:
            raise Exception('Initial cond should have shape: [2, batch_size, hidden_size]\n'
                            'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        return initial_cond

    def __call__(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        assert (isinstance(inputs, list) or isinstance(
            inputs, tuple)) and len(inputs) >= 2
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[ii](
                    self._standardize_condition(c)))
            multi_cond_state = self.multi_cond_p(
                tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.unstack(self.init_state, axis=0)
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out
