import numpy as np
import os.path
from os import mkdir
import shutil
import tensorflow as tf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import RandomOverSampler

from social.data import DataLoader
from social.model import SequentialVAE
from social.plot import plot_results, plot_latents_basic


def main():
    """
    Train model on real data. 
    """

    model_path = os.path.expanduser('~/alspac/social/workspace/tf_vrnn_model')

    load = DataLoader()

    parameters = {
        'x_dim': load.n_variables,
        'h_dim': 16,
        'z_dim': 2,
        'n_time': load.n_timepoints,
        'n_samples': 1,
        'learning_rate': 0.001,
        'batch_size': 32,

        'max_steps': 2000,
        'save_checkpoints_steps': 2000,
        'eval_steps': 1000,
    }

    posterior_parms, prior_parms = train(load.xtr, load.xval, model_path, parameters, clean_model_path=True)
    means, _ = posterior_parms

    style_no = {'color': 'orange', 'alpha': 0.05, 'linewidth': 0.5}
    style_yes = {'color': (0, 109 / 256, 219 / 256), 'alpha': 0.5, 'linewidth': 1.5}
    style_other = {'color': "#000000", 'alpha': 0.5, 'linewidth': 1.5}

    var_factor_codes = {
        'kr812': {'NO': 2, 'YES': 1, 'PDD': -3},
    }

    var_properties = {
        'kr812': {
            'title': 'Disruptive behaviour disorder',
            'style': {'NO': style_no, 'YES': style_yes, 'PDD': style_other},
        },
    }

    plot_results(means, var_factor_codes, var_properties, load.dawba_train, load.dawba_val)


def simulation():
    """
    Train model on simulated data. 
    """

    model_path = os.path.expanduser('~/alspac/social/workspace/tf_vrnn_model_simulated')

    load = DataLoader()

    parameters = {
        'x_dim': load.n_variables,
        'h_dim': 16,
        'z_dim': 2,
        'n_time': load.n_timepoints,
        'n_samples': 1,
        'learning_rate': 0.001,
        'batch_size': 32,

        'max_steps': 2000,
        'save_checkpoints_steps': 2000,
        'eval_steps': 1000,
    }

    z_means, z_covs, x_means, x_covs, zs, xs = load.generate_synthetic(parameters['z_dim'],
                                                                       parameters['h_dim'],
                                                                       8000, missing_prop=0.76)
    xtr, xval = xs
    ztr, zval = zs

    posterior_parms, prior_parms = train(xtr, xval, model_path, parameters, clean_model_path=True)
    means, covs = posterior_parms
    ztr_model, zval_model = means

    plot_latents_basic(ztr, 'simulation/ztr_truth')
    plot_latents_basic(zval, 'simulation/zval_truth')

    plot_latents_basic(ztr_model, 'simulation/ztr_model')
    plot_latents_basic(zval_model, 'simulation/zval_model')


def train(xtr, xval, model_path, parameters, clean_model_path=False):
    """
    Train and evaluate model. Returns approximate posterior means.
    """
    base_path = os.path.expanduser('~/data/alspac/')

    if model_path.startswith(base_path) and clean_model_path:

        shutil.rmtree(model_path)
        mkdir(model_path)

    def train_input_fn():
        return general_input_fn(xtr, parameters['batch_size'], repeat=True, shuffle=True)

    def eval_input_fn_train_set():
        return general_input_fn(xtr, parameters['batch_size'], repeat=False, shuffle=False)

    def eval_input_fn_val_set():
        return general_input_fn(xval, parameters['batch_size'], repeat=False, shuffle=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_path,
        params=parameters,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=parameters['save_checkpoints_steps'],
        ),
    )

    # train model and periodically evaluate
    for _ in range(parameters['max_steps'] // parameters['eval_steps']):
        estimator.train(train_input_fn, steps=parameters['eval_steps'])

        eval_results = estimator.evaluate(eval_input_fn_val_set)

        print("Evaluation_results:\n\t{}\n".format(eval_results))

    # infer prior and posterior parameters on training and test sets
    map_results = map(estimator.predict, [eval_input_fn_train_set, eval_input_fn_val_set])

    qz_mu = map(lambda z: np.array([p['qz_mu'] for p in z]), map_results)
    qz_cov = map(lambda z: np.array([p['qz_cov'] for p in z]), map_results)
    pz_mu = map(lambda z: np.array([p['pz_mu'] for p in z]), map_results)
    pz_cov = map(lambda z: np.array([p['pz_cov'] for p in z]), map_results)

    return (qz_mu, qz_cov), (pz_mu, pz_cov)


def model_fn(features, labels, mode, params, config):
    """
    Model function specifying the model and loss function.
    """
    del labels, config

    x_dim = params['x_dim']
    z_dim = params['z_dim']
    h_dim = params['h_dim']
    n_samples = params['n_samples']
    learning_rate = params['learning_rate']
    n_batch = params['batch_size']
    n_time = params['n_time']
    max_steps = params['max_steps']

    model = SequentialVAE(features, x_dim, z_dim, h_dim, n_time, learning_rate,
                          n_batch, max_steps, n_samples)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'qz_mu': model.qz_mu,
                'qz_cov': model.qz_cov,
                'pz_mu': model.pz_mu,
                'pz_cov': model.pz_cov,
            },
        )

    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=model.loss,
            train_op=model.train_op,
            eval_metric_ops={
                "elbo": tf.metrics.mean(model.elbo),
                "kl_divergence": tf.metrics.mean(model.avg_kl_div),
                "log_prob": tf.metrics.mean(model.avg_log_prob),
            }
        )


def general_input_fn(dataset, batch_size, repeat=False, shuffle=True):

    # mask for NAs:
    mask = np.ma.masked_invalid(dataset)
    mask = mask.mask.astype(int)
    mask = -mask + 1

    # replace NAs with zero
    dataset = np.nan_to_num(dataset)

    # combine mask with dataset
    dataset = np.concatenate([dataset, mask], axis=1)

    dataset = dataset.astype(np.float32)

    if shuffle:
        np.random.shuffle(dataset)

    dataset = tf.data.Dataset.from_tensor_slices(dataset)\
        .map(lambda row: (row, 0))\
        .batch(batch_size, drop_remainder=True)

    if repeat:
        dataset = dataset.repeat()

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':

    # main()
    simulation()
