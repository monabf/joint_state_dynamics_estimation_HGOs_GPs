import os

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from plotting_closedloop_rollouts import model_closedloop_rollout
from plotting_kalman_rollouts import model_kalman_rollout
from plotting_rollouts import model_rollout
from utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1

sb.set_style('whitegrid')


# Some useful plotting functions to make nice graphs of dynamics GPs,
# and other general things such as rollouts, model evaluation


# Evaluate model over a grid
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
# https://stackoverflow.com/questions/36607742/drawing-phase-space-trajectories-with-arrows-in-matplotlib
def model_evaluation(Evaluation_grid, Grid_controls, Predicted_grid,
                     True_predicted_grid, folder, ground_truth_approx=False,
                     title=None, quiver=False, verbose=False):
    nb = 50000
    Evaluation_grid = Evaluation_grid[:nb]
    Grid_controls = Grid_controls[:nb]
    Predicted_grid = Predicted_grid[:nb]
    True_predicted_grid = True_predicted_grid[:nb]
    if not ground_truth_approx:
        quiver = False

    for i in range(Evaluation_grid.shape[1]):
        for j in range(True_predicted_grid.shape[1]):
            if title:
                name = title + str(i) + str(j) + '.pdf'
            else:
                name = 'Model_evaluation' + str(i) + str(j) + '.pdf'
            if quiver:
                plt.quiver(Evaluation_grid[:-1, i], True_predicted_grid[:-1, j],
                           Evaluation_grid[1:, i] - Evaluation_grid[:-1, i],
                           True_predicted_grid[1:, j] -
                           True_predicted_grid[:-1, j],
                           label='True evolution', color='green', alpha=0.9,
                           angles='xy', scale_units='xy', scale=1)
                plt.quiver(Evaluation_grid[:-1, i], Predicted_grid[:-1, j],
                           Evaluation_grid[1:, i] - Evaluation_grid[:-1, i],
                           Predicted_grid[1:, j] - Predicted_grid[:-1, j],
                           label='GP mean prediction', color='blue', alpha=0.6,
                           angles='xy', scale_units='xy', scale=1)
            else:
                plt.scatter(Evaluation_grid[:, i], True_predicted_grid[:, j],
                            s=5, c='g', label='True model', alpha=0.9)
                plt.scatter(Evaluation_grid[:, i], Predicted_grid[:, j],
                            s=5, c='b', label='Prediction', alpha=0.6)
            if not ground_truth_approx:
                plt.title('Predicted and true model evaluation')
            else:
                if title and ('Test' in title):
                    plt.title('Predicted and true model evaluation over '
                              'test data')
                elif title and ('Val' in title):
                    plt.title('Predicted and true model evaluation over '
                              'validation data')
                else:
                    plt.title('Predicted and true model evaluation over '
                              'training data')
            plt.legend()
            plt.xlabel('Evaluation points ' + str(i))
            plt.ylabel('Predicted points ' + str(j))
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')
    for i in range(Grid_controls.shape[1]):
        for j in range(True_predicted_grid.shape[1]):
            if title:
                name = title + str(i + Evaluation_grid.shape[1]) + str(j) + \
                       '.pdf'
            else:
                name = 'Model_evaluation' + str(i + Evaluation_grid.shape[1]) \
                       + str(j) + '.pdf'
            if quiver:
                plt.quiver(Grid_controls[:-1, i], True_predicted_grid[:-1, j],
                           Grid_controls[1:, i] - Grid_controls[:-1, i],
                           True_predicted_grid[1:, j] -
                           True_predicted_grid[:-1, j],
                           label='True evolution', color='green', alpha=0.9,
                           angles='xy', scale_units='xy', scale=1)
                plt.quiver(Grid_controls[:-1, i], Predicted_grid[:-1, j],
                           Grid_controls[1:, i] - Grid_controls[:-1, i],
                           Predicted_grid[1:, j] - Predicted_grid[:-1, j],
                           label='GP mean prediction', color='blue', alpha=0.6,
                           angles='xy', scale_units='xy', scale=1)
            else:
                plt.scatter(Grid_controls[:, i], True_predicted_grid[:, j],
                            s=5, c='g', label='True model', alpha=0.9)
                plt.scatter(Grid_controls[:, i], Predicted_grid[:, j],
                            s=5, c='b', label='Prediction', alpha=0.6)
            if not ground_truth_approx:
                plt.title('Predicted and true model evaluation')
            else:
                if title and ('Test' in title):
                    plt.title('Predicted and true model evaluation over '
                              'test data')
                elif title and ('Val' in title):
                    plt.title('Predicted and true model evaluation over '
                              'validation data')
                else:
                    plt.title('Predicted and true model evaluation over '
                              'training data')
            plt.legend()
            plt.xlabel('Control points ' + str(i))
            plt.ylabel('Predicted points ' + str(j))
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


# Compute list of errors/model quality in rollouts
def run_rollouts(dyn_GP, input_rollout_list, folder, observer=None,
                 observe_data=None, discrete_observer=False,
                 closedloop=False, kalman=False, no_GP_in_observer=False,
                 only_prior=False):
    rollout_RMSE_list = np.zeros((len(input_rollout_list), 1))
    rollout_SRMSE_list = np.zeros((len(input_rollout_list), 1))
    rollout_log_AL_list = np.zeros((len(input_rollout_list), 1))
    rollout_stand_log_AL_list = np.zeros((len(input_rollout_list), 1))
    rollout_list = []
    complete_length = np.sum(
        [len(reshape_pt1(input_rollout_list[i, 2])[1:, :]) for i in
         range(len(input_rollout_list))])
    complete_true_mean = np.zeros((
        complete_length, input_rollout_list[0, 2].shape[1]))
    for i in range(len(input_rollout_list)):
        current_mean = reshape_pt1(input_rollout_list[i, 2])[1:, :]
        current_length = len(current_mean)
        complete_true_mean[i * current_length:(i + 1) * current_length] = \
            current_mean
    mean_test_var = np.linalg.det(np.cov(complete_true_mean.T))
    if reshape_pt1(dyn_GP.scaler_Y.mean_).shape[1] == \
            complete_true_mean.shape[1]:
        mean_vector = reshape_pt1(np.repeat(reshape_pt1(dyn_GP.scaler_Y.mean_),
                                            len(complete_true_mean), axis=0))
        var_vector = reshape_pt1(np.repeat(reshape_pt1(dyn_GP.scaler_Y.var_),
                                           len(complete_true_mean), axis=0))
    else:
        mean_vector = reshape_pt1(np.repeat(reshape_pt1(dyn_GP.scaler_X.mean_),
                                            len(complete_true_mean), axis=0))
        var_vector = reshape_pt1(np.repeat(reshape_pt1(dyn_GP.scaler_X.var_),
                                           len(complete_true_mean), axis=0))

    # Quite slow, parallelize a bit?
    for i in range(len(input_rollout_list)):
        if (i == 0) and (dyn_GP.step > 0):
            save = True
            verbose = dyn_GP.verbose
        else:
            save = False
            verbose = False
        if kalman:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_kalman_rollout(
                dyn_GP=dyn_GP, folder=folder, observer=observer,
                observe_data=observe_data, discrete_observer=discrete_observer,
                init_state=reshape_pt1(input_rollout_list[i, 0]),
                control_traj=reshape_pt1(input_rollout_list[i, 1]),
                true_mean=reshape_pt1(input_rollout_list[i, 2]),
                rollout_length=len(input_rollout_list[i, 1]),
                verbose=verbose, save=save,
                no_GP_in_observer=no_GP_in_observer, only_prior=only_prior)
        elif closedloop:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_closedloop_rollout(
                dyn_GP=dyn_GP, folder=folder, observer=observer,
                observe_data=observe_data,
                init_state=reshape_pt1(input_rollout_list[i, 0]),
                control_traj=reshape_pt1(input_rollout_list[i, 1]),
                true_mean=reshape_pt1(input_rollout_list[i, 2]),
                rollout_length=len(input_rollout_list[i, 1]),
                verbose=verbose, save=save,
                no_GP_in_observer=no_GP_in_observer)
        else:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_rollout(
                dyn_GP=dyn_GP, folder=folder,
                init_state=reshape_pt1(input_rollout_list[i, 0]),
                control_traj=reshape_pt1(input_rollout_list[i, 1]),
                true_mean=reshape_pt1(input_rollout_list[i, 2]),
                rollout_length=len(input_rollout_list[i, 1]),
                verbose=verbose, save=save, only_prior=only_prior)
        SRMSE = RMSE / mean_test_var
        stand_log_likelihood = \
            log_likelihood - log_multivariate_normal_likelihood(
                complete_true_mean, mean_vector, var_vector)
        rollout_RMSE_list[i] = RMSE
        rollout_SRMSE_list[i] = SRMSE
        rollout_log_AL_list[i] = log_likelihood
        rollout_stand_log_AL_list[i] = stand_log_likelihood
        rollout_list.append(
            [predicted_mean, predicted_var, predicted_lowconf,
             predicted_uppconf, RMSE, SRMSE, log_likelihood,
             stand_log_likelihood])

    rollout_list = np.array(rollout_list, dtype=object)
    rollout_RMSE = np.mean(reshape_pt1(rollout_RMSE_list))
    rollout_SRMSE = np.mean(reshape_pt1(rollout_SRMSE_list))
    rollout_log_AL = np.mean(reshape_pt1(rollout_log_AL_list))
    rollout_stand_log_AL = np.mean(reshape_pt1(rollout_stand_log_AL_list))

    return rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
           rollout_stand_log_AL


# Plot raw data received by GP
def save_GP_data(dyn_GP, verbose=False):
    for i in range(dyn_GP.X.shape[1]):
        name = 'GP_input' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_X[:, i], label='GP_X' + str(i))
        plt.title('Visualization of state input data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    for i in range(dyn_GP.X.shape[1], dyn_GP.GP_X.shape[1]):
        name = 'GP_input' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_X[:, i], label='GP_X' + str(i), c='m')
        plt.title('Visualization of control input data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    for i in range(dyn_GP.GP_Y.shape[1]):
        name = 'GP_output' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_Y[:, i], label='GP_Y' + str(i), c='orange')
        plt.title('Visualization of output data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    for i in range(dyn_GP.X.shape[1]):
        for j in range(dyn_GP.Y.shape[1]):
            name = 'GP_data' + str(i) + str(j) + '.pdf'
            plt.scatter(dyn_GP.X[:, i], dyn_GP.Y[:, j], c='c')
            plt.title('Direct visualization of GP data')
            plt.xlabel('GP_X_' + str(i))
            plt.ylabel('GP_Y_' + str(j))
            plt.savefig(os.path.join(dyn_GP.results_folder, name),
                        bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


# Plot GP predictions with control = 0, over a linspace of each input dim
# while keeping all other input dims at 0
def plot_GP(dyn_GP, grid, verbose=False):
    xdim = dyn_GP.X.shape[1]
    udim = dyn_GP.U.shape[1]
    for i in range(xdim + udim):
        dataplot = np.linspace(np.min(grid[:, i]), np.max(grid[:, i]),
                               dyn_GP.nb_plotting_pts)
        data = np.zeros((dyn_GP.nb_plotting_pts, xdim + udim))
        data[:, i] = dataplot
        x = data[:, :xdim]
        u = data[:, xdim:]
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = dyn_GP.predict(x, u)
        true_predicted_mean = predicted_mean.copy()
        for idx, _ in enumerate(true_predicted_mean):
            true_predicted_mean[idx] = reshape_pt1(
                dyn_GP.true_dynamics(reshape_pt1(x[idx]), reshape_pt1(u[idx])))
        df = pd.DataFrame(predicted_mean)
        df.to_csv(os.path.join(dyn_GP.results_folder, 'GP_plot_estim' + str(i)
                               + '.csv'), header=False)
        df = pd.DataFrame(true_predicted_mean)
        df.to_csv(os.path.join(dyn_GP.results_folder, 'GP_plot_true' + str(i)
                               + '.csv'), header=False)
        for j in range(dyn_GP.Y.shape[1]):
            # Plot function learned by GP
            name = 'GP_plot' + str(i) + str(j) + '.pdf'
            plt.plot(data[:, i], true_predicted_mean[:, j],
                     label='True function', c='darkgreen')
            plt.plot(data[:, i], predicted_mean[:, j], label='GP mean',
                     alpha=0.9)
            plt.fill_between(data[:, i], predicted_lowconf[:, j],
                             predicted_uppconf[:, j],
                             facecolor='blue', alpha=0.2)
            if not dyn_GP.ground_truth_approx:
                plt.title('Visualization of GP posterior')
            else:
                plt.title('Visualization of GP posterior over training data')
            plt.legend()
            plt.xlabel('Input state ' + str(i))
            plt.ylabel('GP prediction ' + str(j) + '(x)')
            plt.savefig(os.path.join(dyn_GP.results_folder, name),
                        bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')

    if 'Michelangelo' in dyn_GP.system:
        for i in range(dyn_GP.X.shape[1]):
            dataplot = np.linspace(np.min(grid[:, i]), np.max(grid[:, i]),
                                   dyn_GP.nb_plotting_pts)
            data = np.zeros((dyn_GP.nb_plotting_pts, xdim + udim))
            data[:, i] = dataplot
            x = data[:, :xdim]
            u = data[:, xdim:]
            predicted_mean_deriv, predicted_var_deriv, \
            predicted_lowconf_deriv, predicted_uppconf_deriv = \
                dyn_GP.predict_deriv(x, u)
            df = pd.DataFrame(predicted_mean_deriv)
            df.to_csv(os.path.join(dyn_GP.results_folder, 'GP_plot_deriv' +
                                   str(i) + '.csv'), header=False)
            for j in range(predicted_mean_deriv.shape[1]):
                # Plot derivative of function learned by GP
                name = 'GP_plot_deriv' + str(i) + str(j) + '.pdf'
                plt.plot(x[:, i], predicted_mean_deriv[:, j],
                         label='dGP_' + str(j) + '/dx')
                plt.fill_between(x[:, i], predicted_lowconf_deriv[:, j],
                                 predicted_uppconf_deriv[:, j],
                                 facecolor='blue', alpha=0.2)
                if not dyn_GP.ground_truth_approx:
                    plt.title('Visualization of GP posterior derivative')
                else:
                    plt.title(
                        'Visualization of GP posterior derivative over'
                        'training data')
                plt.legend()
                plt.xlabel('Input state ' + str(i))
                plt.ylabel('GP derivative prediction ' + str(j) + '(x)')
                plt.savefig(os.path.join(dyn_GP.results_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')


# Save data from outside the GP model into its results folder
def save_outside_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    for key, val in data_dic.items():
        df = pd.DataFrame(val)
        df.to_csv(os.path.join(outside_folder, key + '.csv'), header=False)


# Plot data from outside the GP model into its results folder, start function
def plot_outside_data_start(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    for key, val in data_dic.items():
        val = reshape_dim1(val)
        for i in range(val.shape[1]):
            name = key + str(i) + '.pdf'
            plt.plot(val[:, i], label=key + str(i))
            plt.title(key)
            plt.legend()
            plt.xlabel('Time steps')
            plt.ylabel(key)
            plt.savefig(os.path.join(outside_folder, name),
                        bbox_inches='tight')
            plt.close('all')

    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        xtraj = reshape_dim1(data_dic.get('xtraj'))
        xtraj_estim = reshape_dim1(data_dic.get('xtraj_estim'))
        if not xtraj.any():
            return 0, 0, 0, 0, 0
        dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
        for i in range(dimmin):
            name = 'xtraj_xtrajestim_' + str(i) + '.pdf'
            plt.plot(xtraj[:, i], label='True state', c='g')
            plt.plot(xtraj_estim[:, i], label='Estimated state',
                     c='orange', alpha=0.9)
            plt.title('True and estimated position over time')
            plt.legend()
            plt.xlabel('Time steps')
            plt.ylabel('x_' + str(i))
            plt.savefig(os.path.join(outside_folder, name), bbox_inches='tight')
            plt.close('all')

        name = 'Estimation_RMSE_time'
        RMSE = RMS(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin])
        SRMSE = RMSE / np.var(xtraj)
        output_RMSE = RMS(xtraj[:, 0] - xtraj_estim[:, 0])
        output_SRMSE = RMSE / np.var(xtraj[:, 0])
        error = np.sqrt(
            np.mean(np.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]),
                    axis=1))
        error_df = pd.DataFrame(error)
        error_df.to_csv(
            os.path.join(outside_folder, name + '_time.csv'),
            header=False)
        plt.plot(error, 'orange', label='Error')
        plt.title('State estimation RMSE over last cycle')
        plt.xlabel('Time steps')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        return error, RMSE, SRMSE, output_RMSE, output_SRMSE


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time)
def plot_outside_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_RMSE = np.concatenate((
            dyn_GP.observer_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, RMSE]))), axis=0)
        dyn_GP.observer_SRMSE = np.concatenate((
            dyn_GP.observer_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, SRMSE]))), axis=0)

        name = 'Estimation_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.observer_RMSE)
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.observer_RMSE[:, 0], dyn_GP.observer_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.observer_SRMSE)
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.observer_SRMSE[:, 0], dyn_GP.observer_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_RMSE = np.concatenate((
            dyn_GP.output_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_RMSE]))), axis=0)
        dyn_GP.output_SRMSE = np.concatenate((
            dyn_GP.output_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_SRMSE]))), axis=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_RMSE)
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_RMSE[:, 0], dyn_GP.output_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_SRMSE)
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_SRMSE[:, 0], dyn_GP.output_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time),
# but targeted for validation data
def plot_outside_validation_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.validation_folder,
                                      'Data_outside_GP')
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_val_RMSE = np.concatenate((
            dyn_GP.observer_val_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, RMSE]))), axis=0)
        dyn_GP.observer_val_SRMSE = np.concatenate((
            dyn_GP.observer_val_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, SRMSE]))), axis=0)

        name = 'Estimation_RMSE'
        val_RMSE_df = pd.DataFrame(dyn_GP.observer_val_RMSE)
        val_RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                           header=False)
        plt.plot(dyn_GP.observer_val_RMSE[:, 0], dyn_GP.observer_val_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time over validation data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        val_SRMSE_df = pd.DataFrame(dyn_GP.observer_val_SRMSE)
        val_SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                            header=False)
        plt.plot(dyn_GP.observer_val_SRMSE[:, 0],
                 dyn_GP.observer_val_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time over validation data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_val_RMSE = np.concatenate((
            dyn_GP.output_val_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_RMSE]))), axis=0)
        dyn_GP.output_val_SRMSE = np.concatenate((
            dyn_GP.output_val_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_SRMSE]))), axis=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_val_RMSE)
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_val_RMSE[:, 0], dyn_GP.output_val_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_val_SRMSE)
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_val_SRMSE[:, 0], dyn_GP.output_val_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Save outside data into validation results folder
def save_outside_validation_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.validation_folder,
                                      'Data_outside_GP')
    save_outside_data(dyn_GP=dyn_GP, data_dic=data_dic,
                      outside_folder=outside_folder)


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time),
# but targeted for test data
def plot_outside_test_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.test_folder, 'Data_outside_GP')
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_test_RMSE = np.concatenate((
            dyn_GP.observer_test_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, RMSE]))), axis=0)
        dyn_GP.observer_test_SRMSE = np.concatenate((
            dyn_GP.observer_test_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, SRMSE]))), axis=0)

        name = 'Estimation_RMSE'
        test_RMSE_df = pd.DataFrame(dyn_GP.observer_test_RMSE)
        test_RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                            header=False)
        plt.plot(dyn_GP.observer_test_RMSE[:, 0],
                 dyn_GP.observer_test_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time over test data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        test_SRMSE_df = pd.DataFrame(dyn_GP.observer_test_SRMSE)
        test_SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                             header=False)
        plt.plot(dyn_GP.observer_test_SRMSE[:, 0],
                 dyn_GP.observer_test_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time over test data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_test_RMSE = np.concatenate((
            dyn_GP.output_test_RMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_RMSE]))), axis=0)
        dyn_GP.output_test_SRMSE = np.concatenate((
            dyn_GP.output_test_SRMSE, reshape_pt1(np.array([
                dyn_GP.sample_idx, output_SRMSE]))), axis=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_test_RMSE)
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_test_RMSE[:, 0], dyn_GP.output_test_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_test_SRMSE)
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_test_SRMSE[:, 0], dyn_GP.output_test_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Save outside data into test results folder
def save_outside_test_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.test_folder, 'Data_outside_GP')
    save_outside_data(dyn_GP=dyn_GP, data_dic=data_dic,
                      outside_folder=outside_folder)
