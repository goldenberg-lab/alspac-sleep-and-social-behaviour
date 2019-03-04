import pandas as pd
import numpy as np
import os.path


def _factor_code_check(data, variable, _min, _max):
    """
    Checks that a variable's factor codes are within specified range.
    """
    factor_codes = data[variable].unique()
    factor_codes = factor_codes[~ np.isnan(factor_codes)]

    for v in range(_min, _max + 1):
        assert v in factor_codes

    assert len(factor_codes) == _max + 1 - _min


def _split_train_test(df, train_prop):
    """
    Function to split train and test sets (takes twins into account).
    """
    n_train = int(train_prop * len(df))

    idx_mothers = df.index.str[:-1].unique().values
    np.random.shuffle(idx_mothers)

    train_idx, test_idx = idx_mothers[:n_train], idx_mothers[n_train:]

    idx = df.index.str[:-1]

    return df.loc[idx.isin(train_idx)], df.loc[idx.isin(test_idx)]


def _get_missingness_proportion(np_array):

    count = np.isnan(np_array).sum()
    n_elements = np.prod(np_array.shape)
    return count / n_elements


class DataLoader(object):

    def __init__(self):

        paths = ['~/alspac/social/workspace/social_data.csv',
                 '~/alspac/social/workspace/variables.txt',
                 '~/alspac/social/workspace/kr_dawba_91_months.csv']

        _data, _catalogue, _dawba = map(os.path.expanduser, paths)

        self.data = pd.read_csv(_data, index_col=0)
        self.cat = pd.read_csv(_catalogue)
        self.dawba = pd.read_csv(_dawba, index_col=0)

        self._preprocess_cat()
        self._preprocess_data()

        self.timepoints = self.cat.Months.unique()
        self.timepoints.sort()

        self.categories = self.cat.Category.unique().tolist()

        self.train_set, self.val_set = _split_train_test(self.data, train_prop=0.7)

        self.xtr, self.xval = self._convert_to_timeseries()

        self._print_missingness_proportions()

        self.dawba_train = self.dawba.loc[self.dawba.index.isin(self.train_set.index)]
        self.dawba_val = self.dawba.loc[self.dawba.index.isin(self.val_set.index)]

        self.n_timepoints = self.xtr.shape[1]
        self.n_variables = self.xtr.shape[2]
        self.n_train = len(self.xtr)
        self.n_val = len(self.xval)

    def _print_missingness_proportions(self):

        miss_tr = _get_missingness_proportion(self.xtr)
        miss_val = _get_missingness_proportion(self.xval)

        print("Training set, missingness proportion: {}".format(miss_tr))
        print("Validation set, missingness proportion: {}".format(miss_val))

    def _convert_to_timeseries(self):

        variable_positions = dict()

        for i, c in enumerate(self.categories):

            traj = self.cat[self.cat.Category == c]

            for _, entry in traj.iterrows():
                var_id = entry['ID']
                month = entry['Months']
                t = np.where(self.timepoints == month)[0][0]
                variable_positions[var_id] = (t, i)

        n_timepoints = max([t + 1 for t, v in variable_positions.values()])
        n_variables = max([v + 1 for t, v in variable_positions.values()])

        n_train, n_val = len(self.train_set), len(self.val_set)

        xtr = np.empty((n_train, n_timepoints, n_variables))
        xval = np.empty((n_val, n_timepoints, n_variables))

        xtr[...] = np.nan
        xval[...] = np.nan

        for var_id, (t, pos) in variable_positions.items():

            xtr[:, t, pos] = self.train_set[var_id].values
            xval[:, t, pos] = self.val_set[var_id].values

        return xtr, xval

    def _preprocess_cat(self):

        # for now, ignore multiple variables (first check factor codes if you decide to incorporate them)
        self.cat = self.cat.loc[~ self.cat.is_multiple]

        # remove problematic variables (for now)
        to_remove = ['kl715', 'kn6015', 'kj605', 'kl762', 'kj642', 'ta3003']
        self.cat = self.cat.loc[~ self.cat.ID.isin(to_remove)]

    def _preprocess_data(self):

        # remove rows with all NAs
        self.data.dropna(axis=0, how='all', inplace=True)

        self._flip_factor_codes(['kj628', 'kj617', 'kj620', 'kj630', 'kj607'])

        self._scale_factor_codes({'kl730': '5to3', 'kn6020': '5to3',
                                  'kl765': '4to3', 'tb3003': '5to3',
                                  'kj607': '3to5'})

    def _flip_factor_codes(self, to_flip):
        """
        Flip factor codes for input variables, i.e. replaces 1to3 with 3to1.
        """
        for var in to_flip:
            _factor_code_check(self.data, var, _min=1, _max=3)

            self.data[var].replace(1, 0)
            self.data[var].replace(3, 1)
            self.data[var].replace(0, 3)

    def _scale_factor_codes(self, to_scale):
        """
        Scale factor codes. Scale_type must be one of '5to3', '4to3', or '3to5'.
        """
        for var, scale_type in to_scale.items():

            if scale_type == '5to3':

                _factor_code_check(self.data, var, _min=1, _max=5)

                self.data[var] = (((self.data[var] - 1) / 4) * 2) + 1

            elif scale_type == '4to3':

                _factor_code_check(self.data, var, _min=1, _max=4)

                self.data[var] = (((self.data[var] - 1) / 3) * 2) + 1

            elif scale_type == '3to5':

                _factor_code_check(self.data, var, _min=1, _max=3)

                self.data[var].replace(3, 5)
                self.data[var].replace(2, 3)

            else:
                raise Exception

    def generate_synthetic(self, z_dim, h_dim, n_samples, missing_prop=0.0):
        """
        Generates synthetic data using a state space model with ordinal outputs, and Gaussian latents.
        """
        matrix_hz = np.random.rand(z_dim, h_dim) - 0.5
        matrix_zh = np.random.rand(h_dim, z_dim) - 0.5
        matrix_hx = np.random.rand(self.n_variables, h_dim) - 0.5
        matrix_xh = np.random.rand(h_dim, self.n_variables) - 0.5
        matr_h = [np.random.rand(h_dim, h_dim) - 0.5 for _ in range(3)]

        def fn_z(z):
            z = np.matmul(matrix_zh, z)
            return 1.01 * z - 0.5

        def fn_x(x):
            x = np.matmul(matrix_xh, x)
            return 0.8 * x + 1

        def split_mean_cov(v):
            mu = 1.01 * v + 0.1
            cov = np.abs(v + 0.1)
            return mu, cov

        def fn_prior(h):
            z = np.matmul(matrix_hz, h)
            return split_mean_cov(z)

        def fn_dec(z, h):
            z = fn_z(z)
            x = np.matmul(matrix_hx, z + h)
            return split_mean_cov(x)

        def fn_h(x, z, h):
            z = fn_z(z)
            x = fn_x(x)
            z = np.matmul(matr_h[0], z)
            x = np.matmul(matr_h[1], x)
            h = np.matmul(matr_h[2], h)
            return np.tanh(z + x + h)

        z_means, z_covs = [], []
        x_means, x_covs = [], []
        zs, xs = [], []

        for i in range(n_samples):

            if i % 500 == 0:
                print("Generating sample # {}...".format(i+1), flush=True)

            mu_zs, cov_zs = [], []
            mu_xs, cov_xs = [], []
            zts, xts = [], []

            h_prev = np.ones(h_dim)

            for t in range(self.n_timepoints):

                mu_z, cov_z = fn_prior(h_prev)

                z_t = np.random.multivariate_normal(mu_z, np.diag(cov_z))

                mu_x, cov_x = fn_dec(z_t, h_prev)

                x_t = np.random.multivariate_normal(mu_x, np.diag(cov_x))

                h_t = fn_h(x_t, z_t, h_prev)

                zts.append(z_t)
                xts.append(x_t)

                mu_zs.append(mu_z)
                cov_zs.append(cov_z)

                mu_xs.append(mu_x)
                cov_xs.append(cov_x)

                h_prev = h_t

            z_means.append(np.array(mu_zs))
            z_covs.append(np.array(cov_zs))

            x_means.append(np.array(mu_xs))
            x_covs.append(np.array(cov_xs))

            zs.append(np.array(zts))
            xs.append(np.array(xts))

        z_means, z_covs, x_means, x_covs, zs, xs = map(np.array, [z_means, z_covs, x_means, x_covs, zs, xs])

        if missing_prop > 0:

            assert missing_prop < 1
            n_missing = int(xs.size * missing_prop)
            xs.ravel()[np.random.choice(xs.size, n_missing, replace=False)] = np.nan

        train_prop = 0.7
        n_train = int(train_prop * n_samples)

        def split(data):
            return data[:n_train], data[n_train:]

        return map(split, [z_means, z_covs, x_means, x_covs, zs, xs])













