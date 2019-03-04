"""
This code is based on https://github.com/phreeza/tensorflow-vrnn/blob/master/model_vrnn.py
"""
import tensorflow as tf
import tensorflow_probability as tfp


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "linear"):

        w = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))

        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, w) + b, w, b

        else:
            return tf.matmul(input_, w) + b


class VRNNCell(tf.contrib.rnn.RNNCell):
    """
    Variational RNN Cell.

    Modified to accomodate missing data.
    """
    def __init__(self, x_dim, h_dim, z_dim, n_samples=1):

        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim

        self.n_x_1 = x_dim
        self.n_z_1 = z_dim

        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim

        self.n_samples = n_samples if n_samples > 1 else ()

        self.rnn_cell = tf.contrib.rnn.GRUBlockCellV2(num_units=self.n_h)

    @property
    def state_size(self):

        return self.n_h

    @property
    def output_size(self):

        return self.n_h

    def __call__(self, x, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):

            h = state
            assert not isinstance(h, tuple)

            with tf.variable_scope("prior"):

                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(h, self.n_prior_hidden))

                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)

                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))

            with tf.variable_scope("phi_x"):

                x_1 = tf.nn.relu(linear(x, self.n_x_1))  # This step is optional...

            with tf.variable_scope("encoder"):

                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(axis=1, values=(x_1, h)), self.n_enc_hidden))

                with tf.variable_scope("mu"):
                    enc_mu = linear(enc_hidden, self.n_z)

                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.n_z))

            eps = tf.random.normal(enc_sigma.shape, 0.0, 1.0, dtype=tf.float32)

            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))

            with tf.variable_scope("phi_z"):

                z_1 = tf.nn.relu(linear(z, self.n_z_1))   # This step is optional...

            with tf.variable_scope("decoder"):

                with tf.variable_scope("hidden"):
                    dec_hidden = tf.nn.relu(linear(tf.concat(axis=1, values=(z_1, h)), self.n_dec_hidden))

                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.n_x)

                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.n_x))

            # call RNN cell (1 step)
            _, h_new = self.rnn_cell(inputs=tf.concat(axis=1, values=(x_1, z_1)), state=state)

        return (enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma), h_new









