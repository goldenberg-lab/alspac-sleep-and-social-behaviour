"""
This model is based on combining the following two papers:

https://arxiv.org/abs/1506.02216
https://arxiv.org/abs/1807.03653
"""
import tensorflow as tf
import tensorflow_probability as tfp

from social.vrnn_cell import VRNNCell


class SequentialVAE(object):

    def __init__(self, features, x_dim, z_dim, h_dim, n_time,
                 learning_rate, n_batch, max_steps, n_samples=1):

        x = features[:, :n_time, :]  # actual data  (n_batch, n_time, n_dim)
        mask = features[:, n_time:, :]  # missingness mask

        cell = VRNNCell(x_dim, h_dim, z_dim, n_samples)

        h_initial = cell.zero_state(batch_size=n_batch, dtype=tf.float32)

        outputs, h_final = tf.nn.static_rnn(cell=cell, inputs=tf.unstack(x, axis=1), dtype=tf.float32,
                                            initial_state=h_initial)

        outputs_reshape = []
        names = ['enc_mu', 'enc_sigma', 'dec_mu', 'dec_sigma', 'prior_mu', 'prior_sigma']

        for n, name in enumerate(names):
            with tf.variable_scope(name):
                v = tf.stack([o[n] for o in outputs])
                outputs_reshape.append(v)

        enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma = outputs_reshape

        self.z_rvs = []
        x_rvs = []
        self.z_prior_rvs = []

        for t in range(len(outputs)):
            with tf.variable_scope('z_{}'.format(t)):
                zrv = tfp.distributions.MultivariateNormalDiag(loc=enc_mu[t],
                                                               scale_diag=enc_sigma[t])
                self.z_rvs.append(zrv)

            with tf.variable_scope('x_{}'.format(t)):
                xrv = tfp.distributions.Normal(loc=dec_mu[t], scale=dec_sigma[t])
                x_rvs.append(xrv)

            with tf.variable_scope('z_prior_{}'.format(t)):
                z_prior_rv = tfp.distributions.MultivariateNormalDiag(loc=prior_mu[t],
                                                                      scale_diag=prior_sigma[t])
                self.z_prior_rvs.append(z_prior_rv)

        log_probs = [x_t.log_prob(x[:, j, :]) for j, x_t in enumerate(x_rvs)]

        masks = tf.unstack(mask, axis=1)

        log_probs = [tf.multiply(lp, m) for lp, m in zip(log_probs, masks)]

        log_probs = [tf.reduce_sum(l, axis=1) for l in log_probs]

        log_prob = tf.stack(log_probs, axis=1)

        log_prob = tf.reduce_sum(log_prob, axis=1)

        self.avg_log_prob = tf.reduce_mean(log_prob)
        tf.summary.scalar("log_prob", self.avg_log_prob)

        kl_divs = [tfp.distributions.kl_divergence(q_z, p_z)
                   for q_z, p_z in zip(self.z_rvs, self.z_prior_rvs)]

        kl_div = tf.stack(kl_divs, axis=1)

        kl_div = tf.reduce_sum(kl_div, axis=1)

        self.avg_kl_div = tf.reduce_mean(kl_div)
        tf.summary.scalar("kl_divergence", self.avg_kl_div)

        global_step = tf.train.get_or_create_global_step()

        kl_anneal = tf.constant(1, dtype=tf.float32)
        kl_anneal = 1 - tf.train.cosine_decay(kl_anneal, global_step, max_steps)
        tf.summary.scalar("kl_anneal", kl_anneal)

        self.elbo = tf.reduce_mean(log_prob - (kl_anneal * kl_div))
        tf.summary.scalar("elbo", self.elbo)

        self.loss = -self.elbo

        learning_rate = tf.train.cosine_decay(learning_rate, global_step, max_steps)
        tf.summary.scalar("learning_rate", learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        self.qz_mu = tf.transpose(enc_mu, [1, 0, 2])
        self.qz_cov = tf.transpose(enc_sigma, [1, 0, 2])

        self.pz_mu = tf.transpose(prior_mu, [1, 0, 2])
        self.pz_cov = tf.transpose(prior_sigma, [1, 0, 2])
