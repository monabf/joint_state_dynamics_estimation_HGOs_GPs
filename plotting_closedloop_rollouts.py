import logging
import os

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from observers import dynamics_traj_observer
from utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1, interpolate

sb.set_style('whitegrid')


# Some useful plotting functions to run closed-loop rollouts (estimated
# trajectory given noisy measurements and a GP model)


# Run rollouts also in closed loop, i.e. just estimating xhat(t) depending on
# y(t) with the current GP model
def model_closedloop_rollout(dyn_GP, observer, observe_data, folder, init_state,
                             control_traj, true_mean, rollout_length=100,
                             title=None, verbose=False, save=False,
                             no_GP_in_observer=False):
    rollout_length = int(np.min([rollout_length, len(true_mean) - 1]))
    time = np.arange(0, rollout_length + 1) * dyn_GP.dt
    kwargs = dyn_GP.config
    kwargs.update(dict(closedloop=True))
    if no_GP_in_observer:
        GP = dyn_GP.observer_prior_mean
    else:
        GP = dyn_GP

    t_u = np.concatenate((reshape_dim1(time[:-1]), reshape_dim1(control_traj)),
                         axis=1)
    controller = lambda t, kwarg, t_init, u_init: \
        interpolate(t, t_u, t0=time[0], init_value=reshape_pt1(control_traj[0]),
                    method='linear')
    y_observed = reshape_dim1(observe_data(true_mean))
    if 'noisy_inputs' in dyn_GP.system:
        y_observed = reshape_pt1(y_observed + np.random.normal(0, np.sqrt(
            dyn_GP.true_meas_noise_var), y_observed.shape))
    init_state_estim = np.concatenate((reshape_pt1(y_observed[0]), np.zeros((
        1, init_state.shape[1] - 1))), axis=1)  # xhat0 = (x0_1 + noise,0,...,0)
    if 'Michelangelo' in dyn_GP.system:
        init_state_estim = np.concatenate((init_state_estim, reshape_pt1([0])),
                                          axis=1)  # initial guess of xi = 0
    t_y = np.concatenate((reshape_dim1(time), reshape_dim1(y_observed)),
                         axis=1)
    measurement = lambda t, kwarg: interpolate(
        t, t_y, t0=time[0], init_value=init_state_estim[:, 0], method='linear')
    if ('No_observer' in dyn_GP.system) or ('observer' not in
                                            dyn_GP.system):
        predicted_mean = true_mean
        logging.warning('No observer has been specified: the closed-loop '
                        'rollouts are simply the true trajectories.')
    else:
        predicted_mean = dynamics_traj_observer(
            x0=reshape_pt1(init_state_estim), u=controller, y=measurement,
            t0=time[0], dt=dyn_GP.dt, init_control=control_traj[0],
            discrete=dyn_GP.discrete, version=observer,
            method=dyn_GP.optim_method, t_span=[time[0], time[-1]],
            t_eval=time, GP=GP, kwargs=kwargs)
    if 'Michelangelo' in dyn_GP.system:
        predicted_mean = predicted_mean[:, :-1]  # get rid of xi traj
    predicted_lowconf = predicted_mean
    predicted_uppconf = predicted_mean
    predicted_var = np.zeros((rollout_length + 1, 1))
    time = np.arange(0, len(true_mean))
    if save:
        for i in range(predicted_mean.shape[1]):
            if title:
                name = title + 'closedloop_rollout_model_predictions' + str(i) \
                       + '.pdf'
            else:
                name = 'Closedloop_rollout_model_predictions' + str(i) + '.pdf'
            plt.plot(time, true_mean[:, i], 'g', label='True trajectory')
            plt.plot(time, predicted_mean[:, i], label='Estimated trajectory',
                     c='orange', alpha=0.9)
            plt.fill_between(time,
                             predicted_lowconf[:, i],
                             predicted_uppconf[:, i],
                             facecolor='orange', alpha=0.2)
            if not dyn_GP.ground_truth_approx:
                plt.title(
                    'Closedloop roll out of predicted and true trajectory '
                    'over time, random start, random control')
            else:
                if title and ('Test' in title):
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory over time over testing data')
                elif title and ('Val' in title):
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory over time over validation data')
                else:
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory over time, random start, data '
                              'control')
            plt.legend()
            plt.xlabel('Time steps')
            plt.ylabel('State')
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')

        for i in range(predicted_mean.shape[1] - 1):
            if title:
                name = title + 'closedloop_rollout_phase_portrait' + str(
                    i) + '.pdf'
            else:
                name = 'Closedloop_rollout_phase_portrait' + str(i) + '.pdf'
            plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                     label='True trajectory')
            plt.plot(predicted_mean[:, i], predicted_mean[:, i + 1],
                     label='Estimated trajectory', c='orange', alpha=0.9)
            plt.fill_between(predicted_mean[:, i],
                             predicted_lowconf[:, i + 1],
                             predicted_uppconf[:, i + 1],
                             facecolor='orange', alpha=0.2)
            if not dyn_GP.ground_truth_approx:
                plt.title('Closedloop roll out of predicted and true phase '
                          'portrait, random start, random control')
            else:
                if title and ('Test' in title):
                    plt.title('Roll out of predicted and true phase portrait '
                              'over testing data')
                elif title and ('Val' in title):
                    plt.title('Closedloop roll out of predicted and true phase '
                              'portrait over validation data')
                else:
                    plt.title('Closedloop roll out of predicted and true phase '
                              'portrait, random start, data control')
            plt.legend()
            plt.xlabel('x_' + str(i))
            plt.ylabel('x_' + str(i + 1))
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')

    RMSE = RMS(predicted_mean - true_mean)
    log_likelihood = log_multivariate_normal_likelihood(true_mean[1:, :],
                                                        predicted_mean[1:, :],
                                                        predicted_var[1:, :])

    return init_state, control_traj, true_mean, predicted_mean, predicted_var, \
           predicted_lowconf, predicted_uppconf, RMSE, log_likelihood


# Save the results of closed-loop rollouts (different nnames)
def save_closedloop_rollout_variables(results_folder, nb_rollouts,
                                      rollout_list, step,
                                      ground_truth_approx=False, plots=True,
                                      title=None):
    if title:
        folder = os.path.join(results_folder, title + '_' + str(step))
    else:
        folder = os.path.join(results_folder, 'Rollouts_' + str(step))
    os.makedirs(folder, exist_ok=True)
    for i in range(nb_rollouts):
        rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
        filename = 'Predicted_mean_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 3]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_var_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 4]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_lowconf_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 5]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_uppconf_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 6]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'RMSE_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 7]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'SRMSE_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 8]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Log_likelihood_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 9]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Standardized_log_likelihood_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(np.array(rollout_list)[i, 10]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        true_mean = reshape_dim1(np.array(rollout_list)[i, 2])
        predicted_mean = reshape_dim1(np.array(rollout_list)[i, 3])
        predicted_lowconf = reshape_dim1(np.array(rollout_list)[i, 5])
        predicted_uppconf = reshape_dim1(np.array(rollout_list)[i, 6])
        time = np.arange(0, len(true_mean))
        if plots:
            for k in range(predicted_mean.shape[1]):
                name = 'Closedloop_rollout_model_predictions' + str(k) + '.pdf'
                plt.plot(time, true_mean[:, k], 'g', label='True trajectory')
                plt.plot(time, predicted_mean[:, k],
                         label='Estimated trajectory', c='orange', alpha=0.9)
                plt.fill_between(time,
                                 predicted_lowconf[:, k],
                                 predicted_uppconf[:, k],
                                 facecolor='orange', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory '
                              'over time, random start, random control')
                else:
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory '
                              'over time, random start, data control')
                plt.legend()
                plt.xlabel('Time steps')
                plt.ylabel('State')
                plt.savefig(os.path.join(rollout_folder, name),
                            bbox_inches='tight')
                plt.close('all')

            for k in range(predicted_mean.shape[1] - 1):
                name = 'Closedloop_rollout_phase_portrait' + str(k) + '.pdf'
                plt.plot(true_mean[:, k], true_mean[:, k + 1], 'g',
                         label='True trajectory')
                plt.plot(predicted_mean[:, k], predicted_mean[:, k + 1],
                         label='Estimated trajectory', c='orange', alpha=0.9)
                plt.fill_between(predicted_mean[:, k],
                                 predicted_lowconf[:, k + 1],
                                 predicted_uppconf[:, k + 1],
                                 facecolor='orange', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Closedloop roll out of predicted and true '
                              'phase portrait over time, random start, '
                              'random control')
                else:
                    plt.title('Closedloop roll out of predicted and true '
                              'phase portrait over time, random start, '
                              'data control')
                plt.legend()
                plt.xlabel('x_' + str(k))
                plt.ylabel('x_' + str(k + 1))
                plt.savefig(os.path.join(rollout_folder, name),
                            bbox_inches='tight')
                plt.close('all')


# Plot quantities about closed-loop rollouts over time
def plot_closedloop_rollout_data(dyn_GP, folder):
    name = 'Closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.closedloop_rollout_RMSE)
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_RMSE[:, 0],
             dyn_GP.closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over '
              + str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.closedloop_rollout_SRMSE)
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_SRMSE[:, 0],
             dyn_GP.closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over ' + str(
        dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.closedloop_rollout_log_AL)
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_log_AL[:, 0],
             dyn_GP.closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.closedloop_rollout_stand_log_AL)
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')


# Plot quantities about test rollouts over time
def plot_test_closedloop_rollout_data(dyn_GP, folder):
    name = 'Test_closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_RMSE)
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_RMSE[:, 0],
             dyn_GP.test_closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_SRMSE)
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_SRMSE[:, 0],
             dyn_GP.test_closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_log_AL)
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_log_AL[:, 0],
             dyn_GP.test_closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_stand_log_AL)
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.test_closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')


# Plot quantities about validation rollouts over time
def plot_val_closedloop_rollout_data(dyn_GP, folder):
    name = 'Val_closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_RMSE)
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_RMSE[:, 0],
             dyn_GP.val_closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_SRMSE)
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_SRMSE[:, 0],
             dyn_GP.val_closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_log_AL)
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_log_AL[:, 0],
             dyn_GP.val_closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_stand_log_AL)
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.val_closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')
