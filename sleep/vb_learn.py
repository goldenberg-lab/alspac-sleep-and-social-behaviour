import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pykalman import KalmanFilter
import pystan


def load_data():
    # file paths
    sleep_path = '~/alspac/sleep/workspace/sleep.csv'
    time_path = '~/alspac/sleep/workspace/times.csv'
    sleep_path, time_path = map(os.path.expanduser, [sleep_path, time_path])

    # read data
    sleep = pd.read_csv(sleep_path)
    times = pd.read_csv(time_path)

    return sleep, times


def prepare(sleep):
    # list variables by type
    categoricals = ['kn2034', 'cct5141', 'cct5142']
    ordinals = ['kk470', 'kk480', 'kn2010', 'kn2122', 'kq280', 'kr204', 'kr206',
                'kr293', 'ku762', 'kv4023', 'kv4024', 'kv4025', 'kv5538', 'kv6555',
                'tb4023', 'tb4024', 'tb4025', 'tb5538', 'tb6555']
    reals = ['kn2011a', 'kq317']  # respectively: 'continuous time in hours', 'positive unbounded integer'
    binaries = ['kn2000', 'kr204a', 'kr206a', 'kr293a', 'kr413', 'kv7034', 'tb7034',
                'cct5140', 'CCU2050c', 'CCU3430', 'CCU3431', 'CCU3432']

    # change categoricals dtype
    sleep = sleep.astype({'kn2034': 'object', 'cct5141': 'object', 'cct5142': 'object'})

    # for now, only use ordinals and reals
    sleep = sleep[ordinals + reals]

    # remove completely missing rows
    sleep.dropna(axis=0, how='all', inplace=True)

    return sleep


def temporary_steps(sleep):
    # for now, drop certain columns with high missingness
    sleep.dropna(axis=1, thresh=7000, inplace=True)

    # for now, remove all rows with any missing data
    sleep.dropna(axis=0, how='any', inplace=True)

    return sleep


def timeseries(sleep, times):
    # get subset of timepoints captured in data
    sub = times[times.variable.isin(sleep.columns)]
    t_pts = sub.time.unique()

    # create np.array at each time point
    arrays = []

    for t in sorted(t_pts):

        # observed and missing variables
        observed = sub[sub.time == t].variable
        missing = sub[sub.time != t].variable

        # dataframe for current timepoint
        current = sleep.copy()

        # mask all variables not in current timepoint
        current[missing] = np.nan

        # synonyms (variables that refer to the same thing)
        syns = [['kr206', 'kv4025'], ['kr204', 'kv4023'], ['kn2122', 'ku762']]

        # combine synonym variables
        for ls in syns:
            orig = ls.pop(0)
            for copy in ls:

                # sanity check (makes sure we're not erasing any data)
                if not (current[copy].isna().all() or current[orig].isna().all()):
                    raise Exception

                # copy data over
                if current[orig].isna().all():
                    current[orig] = current[copy]

                # remove duplicate column
                current.drop(labels=copy, axis=1, inplace=True)

        # to np.array
        current = current.values

        arrays.append(current)

    return np.stack(arrays, axis=1)


def kf_learn(sleep):
    # dimensionality
    D = sleep.shape[2]

    # mask all nans
    sleep = np.ma.masked_invalid(sleep)

    # list of models (for each individual)
    models, latents = [], []

    for i, x in enumerate(self.sleep):

        if i % 250 == 0:
            print("At example: {}".format(i))

        # declare kalman filter object
        kf = KalmanFilter(n_dim_state=1, n_dim_obs=self.D)

        # em steps
        kf.em(x, n_iter=10, em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance',
                                     'observation_covariance', 'initial_state_mean', 'initial_state_covariance'])

        # filter
        means, covs = kf.filter(x)
        result = (means, covs)

        # save
        models.append(kf)
        latents.append(result)

    return models, latents


def process_stan(sleep, L=1):
    # sleep: N x T x D np.array

    _, T, D = sleep.shape

    data = []

    for series in sleep:

        x_obs, ii_obs, ii_mis = [], [], []
        s, m, pos_obs, pos_mis = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        for t, x in enumerate(series):

            x_obs.extend(x[~np.isnan(x)].tolist())

            idcs = np.argwhere(~np.isnan(x))
            idcs = np.squeeze(idcs, axis=1) + 1
            ii_obs.extend(idcs.tolist())

            s[t] = len(idcs)

            if t == 0:
                pos_obs[t] = 1;
            else:
                pos_obs[t] = pos_obs[t - 1] + s[t - 1]

            idcs = np.argwhere(np.isnan(x))
            idcs = np.squeeze(idcs, axis=1) + 1
            ii_mis.extend(idcs.tolist())

            m[t] = len(idcs)

            if t == 0:
                pos_mis[t] = 1;
            else:
                pos_mis[t] = pos_mis[t - 1] + m[t - 1]

        N_obs = len(ii_obs)
        N_mis = len(ii_mis)

        m = m.astype(int)
        s = s.astype(int)
        pos_obs = pos_obs.astype(int)
        pos_mis = pos_mis.astype(int)

        data_dict = {
            'T': T, 'D': D, 'L': L, 'N_obs': N_obs, 'N_mis': N_mis,
            'x_obs': x_obs, 's': s, 'm': m, 'pos_obs': pos_obs, 'pos_mis': pos_mis,
            'ii_obs': ii_obs, 'ii_mis': ii_mis,
        }

        data.append(data_dict)

    return data


def synthetic(scale=1.0, offset=0.0, mask_thresh=0.0):
    T, D, L = (9, 5, 1)
    X = (np.random.rand(T, D) * scale) + offset

    mis = np.random.rand(T, D) - mask_thresh
    mis = np.round(mis).astype(bool)
    X[mis] = np.nan

    x_obs, ii_obs, ii_mis = [], [], []
    s, m, pos_obs, pos_mis = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

    for t, x in enumerate(X):

        x_obs.extend(x[~np.isnan(x)].tolist())

        idcs = np.argwhere(~np.isnan(x))
        idcs = np.squeeze(idcs, axis=1) + 1
        ii_obs.extend(idcs.tolist())

        s[t] = len(idcs)

        if t == 0:
            pos_obs[t] = 1;
        else:
            pos_obs[t] = pos_obs[t - 1] + s[t - 1]

        idcs = np.argwhere(np.isnan(x))
        idcs = np.squeeze(idcs, axis=1) + 1
        ii_mis.extend(idcs.tolist())

        m[t] = len(idcs)

        if t == 0:
            pos_mis[t] = 1;
        else:
            pos_mis[t] = pos_mis[t - 1] + m[t - 1]

    N_obs = len(ii_obs)
    N_mis = len(ii_mis)

    m = m.astype(int)
    s = s.astype(int)
    pos_obs = pos_obs.astype(int)
    pos_mis = pos_mis.astype(int)

    data = {
        'T': T, 'D': D, 'L': L, 'N_obs': N_obs, 'N_mis': N_mis,
        'x_obs': x_obs, 's': s, 'm': m, 'pos_obs': pos_obs, 'pos_mis': pos_mis,
        'ii_obs': ii_obs, 'ii_mis': ii_mis,
    }

    return data


def stan_kf():
    return pystan.StanModel(file='program_miss.stan')


# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


if __name__ == '__main__':

    sleep, times = load_data()
    sleep = prepare(sleep)
    sleep = temporary_steps(sleep)

    multi_timeseries = timeseries(sleep, times)
    T = multi_timeseries.shape[1]

    dicts = process_stan(multi_timeseries, L=1)

    model = stan_kf()

    n_examples = len(dicts)
    sample_size = 1000

    # sample = np.random.randint(0, n_examples, sample_size)
    sample = np.arange(0, n_examples)

    posterior_est = []

    for i, s in enumerate(sample):

        # data_dict = synthetic(3, 1.5, 1)

        print("At sample: {}".format(i), flush=True)

        data_dict = dicts[s]

        try:
            with suppress_stdout_stderr():
                res = model.vb(data=data_dict, iter=1000)
                res = {k: v for k, v in zip(res['mean_par_names'], res['mean_pars'])}

        except Exception as e:
            print(e)
            continue

        y = []
        for j in range(T):
            y.append(res['y[' + str(j + 1) + ',1]'])

        y = np.array(y)

        y = np.expand_dims(y, axis=1)

        posterior_est.append(y)

    means = np.concatenate(posterior_est, axis=1)

    T, N = means.shape
    df = pd.DataFrame(means)

    sns.set(style="whitegrid")
    sns.set(color_codes=True)

    plt.figure(figsize=(10, 7))

    if plt.axes().get_legend():
        plt.axes().get_legend().remove()

    y_mean = df.mean(axis=0)
    y_min, y_max = y_mean.min(), y_mean.max()

    for column in df:
        plt.plot(df.index, df[column], c=sns.color_palette()[2], alpha=.2)

    plt.savefig('output_vb.png')
