import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
plt.style.use('ggplot')
plt.style.use('seaborn-poster')


def plot_results(means, var_factor_codes, var_properties, dawba_train, dawba_val):
    """
    Plot latents for all relevant outcomes.
    """
    ztr, zval = means

    for code, factors in var_factor_codes.items():
        print("Plotting {}".format(code), flush=True)

        title = var_properties[code]['title']
        style = var_properties[code]['style']

        _plot_latents(ztr, code, title, factors, style, dawba_train, base_name='ztr_')

        _plot_latents(zval, code, title, factors, style, dawba_val, base_name='zval_')


def _plot_latents(z, outcome_var_name, outcome_var_title, factor_codes, factor_styles,
                  outcome_set, base_name='ztr_', suffix=None):
    """
    Plots over each latent dimension, train/test sets, and various DAWBA codes.

    Example input:

    z has dimensions (n_examples, n_timepoints, n_dim)

    outcome_var_name = 'kr812'

    outcome_set = dawba_train

    factor_codes = {
        'normal': 2,
        'diagnosed': 1,
        'pdd': -3,
    }

    factor_styles = {
        'normal': {'color':'green', 'alpha':0.05, 'linewidth':0.5},
        'diagnosed': {'color':'blue', 'alpha':0.5, 'linewidth':1.5},
        'pdd': {'color':'red', 'alpha':0.5, 'linewidth':1.5},
    }
    """
    n_examples = z.shape[0]
    z_dim = z.shape[2]

    outcome = outcome_set[outcome_var_name].values
    outcome = outcome[:n_examples]

    code_dict = {}
    for k, v in factor_codes.items():
        code_dict[k] = outcome == v

    legend = [mpatches.Patch(color=val['color'], label=key) for key, val in factor_styles.items()]

    for j in range(z_dim):

        plt.figure(figsize=(20, 20))

        for i, traj in enumerate(z):

            for _code, vals in code_dict.items():

                _style = factor_styles[_code]

                if vals[i]:

                    plt.plot(range(len(traj)), traj[:, j], color=_style['color'],
                             alpha=_style['alpha'], linewidth=_style['linewidth'])
                    break

        plt.title(outcome_var_title)
        plt.legend(handles=legend)

        fig_path = 'plots/' + base_name
        if suffix:
            fig_path += '_' + suffix + '_'
        fig_path += str(j + 1) + '_' + outcome_var_name

        plt.savefig(fig_path, dpi=300)


def plot_latents_basic(z, plot_name, suffix=None):

    z_dim = z.shape[2]

    for j in range(z_dim):

        plt.figure(figsize=(20, 20))

        for i, traj in enumerate(z):
            plt.plot(range(len(traj)), traj[:, j], color="#000000", alpha=0.05, linewidth=0.5)

        plt.title(plot_name)

        fig_path = 'plots/' + plot_name
        if suffix:
            fig_path += '_' + suffix + '_'
        fig_path += str(j + 1)

        plt.savefig(fig_path, dpi=300)
