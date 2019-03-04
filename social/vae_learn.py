import pandas as pd
import numpy as np
import os.path

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use("ggplot")


# file paths
data_path = os.path.expanduser('~/alspac/social/workspace/social_data.csv')
catalogue_path = os.path.expanduser('~/alspac/social/workspace/variables.txt')
dawba_path = os.path.expanduser('~/alspac/social/workspace/kr_dawba_91_months.csv')
model_path = os.path.expanduser('~/alspac/social/workspace/tf_model/')

# read to dataframes
data = pd.read_csv(data_path, index_col=0)
cat = pd.read_csv(catalogue_path)
dawba = pd.read_csv(dawba_path, index_col=0)


# use KL for now, since it has most number of questions
kl_vars = cat[cat.Months == 57]

# select KL data
kldf = data.loc[:, kl_vars.ID]


# remove rows with all NAs
kldf.dropna(axis=0, how='all', inplace=True)

# filter DAWBA variables correspondingly
dawba = dawba.loc[kldf.index]


# function to split train and test sets (takes twins into account)
def split_train_test(df, train_prop):

    n_train = int(train_prop * len(df))

    idx_mothers = df.index.str[:-1].unique().values
    np.random.shuffle(idx_mothers)

    train_idx, test_idx = idx_mothers[:n_train], idx_mothers[n_train:]

    idx = df.index.str[:-1]

    return df.loc[idx.isin(train_idx)], df.loc[idx.isin(test_idx)]


# split into train and test data
kldf_train, kldf_test = split_train_test(kldf, train_prop=0.7)

# slice DAWBA variables correspondingly
dawba_train = dawba.loc[dawba.index.isin(kldf_train.index)]
dawba_test = dawba.loc[dawba.index.isin(kldf_test.index)]


parameters = {
    'n_inputs': len(kldf.columns),  # 21
    'n_units': 16,
    'n_latents': 2,
    'n_samples': 1,
    'learning_rate': 0.001,
    'batch_size': 32,

    'max_steps': 5000,
    'save_checkpoints_steps': 500,
    'eval_steps': 500,
    'test_batch_size': len(kldf_test)
}


# approximate posterior distribution q(z|x)
def encoder(x, n_units, n_latents):

    encode = keras.Sequential([
        keras.layers.Dense(n_units, activation=tf.nn.relu),
        keras.layers.Dense(2 * n_latents, activation=None),
    ])

    network = encode(x)

    return tfp.distributions.MultivariateNormalDiag(
        loc=network[..., :n_latents],
        scale_diag=tf.nn.softplus(network[..., n_latents:]),  # + tf.log(tf.math.expm1(1.0))),
        name='latent',
    )


# conditional likelihood distribution p(x|z)
def decoder(z, n_units, n_inputs):

    decode = keras.Sequential([
        keras.layers.Dense(n_units, activation=tf.nn.relu),
        keras.layers.Dense(2 * n_inputs, activation=None),
    ])

    network = decode(z)

    '''
    return tfp.distributions.MultivariateNormalDiag(
        loc=network[..., :n_inputs],
        scale_diag=tf.nn.softplus(network[..., n_inputs:]),
        name='observed',
    )
    '''

    return tfp.distributions.Normal(
        loc=network[..., :n_inputs],
        scale=tf.nn.softplus(network[..., n_inputs:]),
        name='observed',
    )


# prior distribution for latent variables p(z)
def latent_prior(n_latents):

    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([n_latents]),
        scale_identity_multiplier=1.0,
    )


# model function which specifies the model and loss
def model_fn(features, labels, mode, params, config):

    del labels, config

    n_units = params['n_units']
    n_inputs = params['n_inputs']
    n_latents = params['n_latents']

    prior = latent_prior(n_latents)

    inputs = features[:, :n_inputs]  # actual data
    mask = features[:, n_inputs:]  # NA mask

    approx_posterior = encoder(inputs, n_units, n_latents)

    posterior_means = approx_posterior.mean(name='posterior_mean')

    approx_posterior_sample = approx_posterior.sample(params['n_samples'])

    decoder_likelihood = decoder(approx_posterior_sample, n_units, n_inputs)

    log_prob = decoder_likelihood.log_prob(inputs)

    # apply NA mask:
    log_prob = tf.squeeze(log_prob)
    log_prob = tf.multiply(log_prob, mask)
    log_prob = tf.reduce_sum(log_prob, axis=1)

    avg_log_prob = tf.reduce_mean(log_prob)
    tf.summary.scalar("log_prob", avg_log_prob)

    kl_div = tfp.distributions.kl_divergence(approx_posterior, prior)

    avg_kl_div = tf.reduce_mean(kl_div)
    tf.summary.scalar("kl_divergence", avg_kl_div)

    elbo = tf.reduce_mean(log_prob - kl_div)
    tf.summary.scalar("elbo", elbo)

    loss = -elbo

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.cosine_decay(params['learning_rate'], global_step,
                                          params['max_steps'])
    tf.summary.scalar("learning_rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'posterior_means': posterior_means,
            },
        )

    else:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={
                "elbo": tf.metrics.mean(elbo),
                "kl_divergence": tf.metrics.mean(avg_kl_div),
                "log_prob": tf.metrics.mean(avg_log_prob),
            }
        )


def general_input_fn(df, batch_size, repeat=False, shuffle=True):

    dataset = df.values

    # mask for NAs:
    mask = np.ma.masked_invalid(dataset)
    mask = mask.mask.astype(int)
    mask = -mask + 1

    # replace NAs with zero
    dataset = np.nan_to_num(dataset)

    # combine with dataset
    dataset = np.concatenate([dataset, mask], axis=1)

    dataset = dataset.astype(np.float32)

    if shuffle:
        np.random.shuffle(dataset)

    dataset = tf.data.Dataset.from_tensor_slices(dataset)\
        .map(lambda row: (row, 0))\
        .batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    return dataset.make_one_shot_iterator().get_next()


def train_input_fn():

    return general_input_fn(kldf_train, parameters['batch_size'], repeat=True, shuffle=True)


def eval_input_fn():

    return general_input_fn(kldf_test, parameters['test_batch_size'], repeat=False, shuffle=False)


estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_path,
    params=parameters,
    config=tf.estimator.RunConfig(
        save_checkpoints_steps=parameters['save_checkpoints_steps'],
    ),
)


for _ in range(parameters['max_steps'] // parameters['eval_steps']):

    estimator.train(train_input_fn, steps=parameters['eval_steps'])

    eval_results = estimator.evaluate(eval_input_fn)

    print("Evaluation_results:\n\t{}\n".format(eval_results))


# get approximate posterior means
preds = estimator.predict(eval_input_fn)  # generator
preds = list(preds)
z_means = [p['posterior_means'] for p in preds]
z_means = np.stack(z_means)


# plot with seaborn:

xaxis, yaxis = z_means[:, 0], z_means[:, 1]

outcome = dawba_test.kr812.values

# 'normal' individuals
normal = outcome == 2
plt.scatter(xaxis[normal], yaxis[normal], alpha=0.1, label='No')

# positive diagnosis
diagnosed = outcome == 1
plt.scatter(xaxis[diagnosed], yaxis[diagnosed], alpha=0.99, label='Yes')

# PDD (pervasive development disorder)
pdd = outcome == -3
plt.scatter(xaxis[pdd], yaxis[pdd], alpha=0.99, label='PDD')

plt.title('Disruptive Behaviour Disorder')
plt.legend()
plt.show()

# kr803a, kr803, kr801, kr827, kr812



