# Copyright 2019 Max Planck Society. All rights reserved.
import logging
import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

# from gym import wrappers
# from matplotlib import animation as anim

sb.set_style('whitegrid')
plt.rcParams.update({'font.size': 22})


def box_plots_vars_time(folder, variables=['RMSE'], avoid0=False,
                        errorbars=False):
    # List all folders = different experiments in current folder
    subfolders_unfiltered = [os.path.join(folder, o) for o in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, o))]
    subfolders = []
    for subfolder in subfolders_unfiltered:
        if ('Box_plots' in subfolder) or ('Ignore' in subfolder):
            continue  # Skip all subfolders containing Box_plots so avoid!
        subfolders += [subfolder]

    # Gather all variables to plot in dictionary
    vars = dict.fromkeys(variables)
    for key, val in vars.items():
        vars[key] = []
        for subfolder in subfolders:
            try:
                name = key + '.csv'
                data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                                   header=None)
                values = data.drop(data.columns[0], axis=1).values
                if values[-1, 1] > 50:
                    # Ignore folders with values to large (must be problem)
                    logging.warning('Experiment ' + str(subfolder) +
                                    ' was ignored because of unexpectedly '
                                    'large error values')
                    subfolders.remove(subfolder)
                    continue
                if avoid0:
                    # Avoid first value at sample_idx = 0
                    if 'Data_outside' not in key:
                        values = values[np.logical_not(values[:, 0] == 0)]
                vars[key].append(values)
            except FileNotFoundError:
                print(
                    'Files ' + os.path.join(subfolder, str(key)) + ' not found')

    # Make results folder
    results_folder = os.path.join(folder, 'Box_plots')
    os.makedirs(results_folder, exist_ok=True)  # Will overwrite!
    specs_path = os.path.join(results_folder, 'Specifications.txt')
    shutil.copy(os.path.join(subfolders[0], 'Loop_0/Specifications.txt'),
                results_folder)
    with open(specs_path, 'a') as f:
        print('\n', file=f)
        print('\n', file=f)
        print('\n', file=f)
        print('Box plots from ' + str(len(subfolders)) + ' experiments',
              file=f)
        print('Experiments not used: ' +
              str(list(set(subfolders_unfiltered) - set(subfolders))), file=f)

    # Compute mean and std for each variable, then plot
    vars_mean = dict.fromkeys(variables)
    vars_std = dict.fromkeys(variables)
    for key, val in vars.items():
        vars_mean[key] = np.mean(np.array(vars[key]), axis=0)
        vars_std[key] = np.std(np.array(vars[key]), axis=0)
        name = key.replace('/', '_') + '.pdf'  # avoid new dirs
        plt.plot(vars_mean[key][:, 0], vars_mean[key][:, 1], 'deepskyblue')
        if errorbars:
            errorevery = 1
            markevery = 1
            plt.errorbar(vars_mean[key][:, 0], vars_mean[key][:, 1],
                         yerr=vars_std[key][:, 1], fmt='o', ls='-',
                         c='deepskyblue', capsize=2, alpha=0.8,
                         errorevery=errorevery, markevery=markevery)
        else:
            plt.fill_between(vars_mean[key][:, 0],
                             vars_mean[key][:, 1] - vars_std[key][:, 1],
                             vars_mean[key][:, 1] + vars_std[key][:, 1],
                             facecolor='deepskyblue', alpha=0.2)
        plt.xlim(xmin=vars_mean[key][0, 0])
        plt.xlabel('Number of samples')
        plt.ylabel('RMSE')
        plt.savefig(os.path.join(results_folder, name), bbox_inches='tight')
        plt.close('all')


def replot_vars_time(subfolder, variables=['RMSE'], avoid0=False):
    # From single subfolder and variables of type [time, values], replot
    # value by time
    variables_plot = dict.fromkeys(variables)
    for key in variables_plot:
        try:
            name = key + '.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            values = data.drop(data.columns[0], axis=1).values
            if avoid0:
                # Avoid first value at sample_idx = 0
                if 'Data_outside' not in key:
                    values = values[np.logical_not(values[:, 0] == 0)]
            name = key + '.pdf'
            plt.plot(values[:, 0], values[:, 1], 'deepskyblue')
            plt.xlim(xmin=values[key][0, 0])
            plt.xlabel('Number of samples')
            plt.ylabel('RMSE')
            plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
            plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + ' not found')


def replot_rollouts(subfolder, plots=[]):
    # From rollout folder name, replot rollout
    variables_plot = dict.fromkeys(plots)
    for key in variables_plot:
        # True traj
        name = key + '/True_traj.csv'
        data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                           header=None)
        true_mean = data.drop(data.columns[0], axis=1).values
        time = np.arange(0, len(true_mean))

        try:
            # Open-loop trajs
            name = key + '/Predicted_mean_traj.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_mean_traj = data.drop(data.columns[0], axis=1).values
            name = key + '/Predicted_uppconf_traj.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_uppconf = data.drop(data.columns[0], axis=1).values
            name = key + '/Predicted_lowconf_traj.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_lowconf = data.drop(data.columns[0], axis=1).values
            for i in range(predicted_mean_traj.shape[1]):
                name = key + 'Rollout_model_predictions' + str(
                    i) + '.pdf'
                plt.plot(time, true_mean[:, i], 'g', label='True')
                plt.plot(time, predicted_mean_traj[:, i],
                         label='Predicted', c='b', alpha=0.7)
                plt.fill_between(time,
                                 predicted_traj_lowconf[:, i],
                                 predicted_traj_uppconf[:, i],
                                 facecolor='blue', alpha=0.2)
                plt.legend()
                # plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5),
                #            frameon=True)
                plt.xlim(xmin=time[0])
                plt.xlabel(r't')
                plt.ylabel(r'$z_{}$'.format(i + 1))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
            for i in range(predicted_mean_traj.shape[1] - 1):
                name = key + 'Rollout_phase_portrait' + str(i) + '.pdf'
                plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                         label='True')
                plt.plot(predicted_mean_traj[:, i],
                         predicted_mean_traj[:, i + 1],
                         label='Predicted', c='b', alpha=0.7)
                plt.fill_between(predicted_mean_traj[:, i],
                                 predicted_traj_lowconf[:, i + 1],
                                 predicted_traj_uppconf[:, i + 1],
                                 facecolor='blue', alpha=0.2)
                plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
                            marker='x', s=100, label='Initial state')
                plt.legend()
                plt.xlabel(r'$z_{}$'.format(i + 1))
                plt.ylabel(r'$z_{}$'.format(i + 2))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + ' openloop '
                                                                 'not found')

        try:
            # Kalman trajs
            name = key + '/Predicted_mean_traj_kalman.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_mean_traj_kalman = data.drop(data.columns[0],
                                                   axis=1).values
            name = key + '/Predicted_uppconf_traj_kalman.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_uppconf_kalman = data.drop(data.columns[0],
                                                      axis=1).values
            name = key + '/Predicted_lowconf_traj_kalman.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_lowconf_kalman = data.drop(data.columns[0],
                                                      axis=1).values
            for i in range(predicted_mean_traj_kalman.shape[1]):
                name = key + 'Kalman_rollout_model_predictions' + str(
                    i) + '.pdf'
                plt.plot(time, true_mean[:, i], 'g', label='True')
                plt.plot(time, predicted_mean_traj_kalman[:, i],
                         label='Predicted', c='b', alpha=0.7)
                plt.fill_between(time,
                                 predicted_traj_lowconf_kalman[:, i],
                                 predicted_traj_uppconf_kalman[:, i],
                                 facecolor='blue', alpha=0.2)
                plt.legend()
                plt.xlim(xmin=time[0])
                plt.xlabel(r't')
                plt.ylabel(r'$z_{}$'.format(i + 1))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
            for i in range(predicted_mean_traj_kalman.shape[1] - 1):
                name = key + 'Kalman_rollout_phase_portrait' + str(i) + '.pdf'
                plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                         label='True')
                plt.plot(predicted_mean_traj_kalman[:, i],
                         predicted_mean_traj_kalman[:, i + 1],
                         label='Predicted', c='b', alpha=0.7)
                plt.fill_between(predicted_mean_traj_kalman[:, i],
                                 predicted_traj_lowconf_kalman[:, i + 1],
                                 predicted_traj_uppconf_kalman[:, i + 1],
                                 facecolor='blue', alpha=0.2)
                plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
                            marker='x', s=100, label='Initial state')
                plt.legend()
                plt.xlim(xmin=time[0])
                plt.xlabel(r'$z_{}$'.format(i + 1))
                plt.ylabel(r'$z_{}$'.format(i + 2))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + ' Kalman not '
                                                                 'found')

        try:
            # Closed-loop trajs
            name = key + '/Predicted_mean_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_mean_traj_closedloop = data.drop(data.columns[0],
                                                       axis=1).values
            name = key + '/Predicted_uppconf_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_uppconf_closedloop = data.drop(data.columns[0],
                                                          axis=1).values
            name = key + '/Predicted_mean_traj_closedloop.csv'
            data = pd.read_csv(os.path.join(subfolder, name), sep=',',
                               header=None)
            predicted_traj_lowconf_closedloop = data.drop(data.columns[0],
                                                          axis=1).values
            for i in range(predicted_mean_traj_closedloop.shape[1]):
                name = key + 'Closedloop_rollout_model_predictions' + str(
                    i) + '.pdf'
                plt.plot(time, true_mean[:, i], 'g', label='True')
                plt.plot(time, predicted_mean_traj_closedloop[:, i],
                         label='Estimated',
                         c='orange', alpha=0.9)
                plt.fill_between(time,
                                 predicted_traj_lowconf_closedloop[:, i],
                                 predicted_traj_uppconf_closedloop[:, i],
                                 facecolor='orange', alpha=0.2)
                plt.legend()
                plt.xlim(xmin=time[0])
                plt.xlabel(r't')
                plt.ylabel(r'$z_{}$'.format(i + 1))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
            for i in range(predicted_mean_traj_closedloop.shape[1] - 1):
                name = key + 'Closedloop_rollout_phase_portrait' + str(
                    i) + '.pdf'
                plt.plot(true_mean[:, i], true_mean[:, i + 1], 'g',
                         label='True')
                plt.plot(predicted_mean_traj_closedloop[:, i],
                         predicted_mean_traj_closedloop[:, i + 1],
                         label='Estimated', c='orange', alpha=0.9)
                plt.fill_between(predicted_mean_traj_closedloop[:, i],
                                 predicted_traj_lowconf_closedloop[:, i + 1],
                                 predicted_traj_uppconf_closedloop[:, i + 1],
                                 facecolor='orange', alpha=0.2)
                plt.scatter(true_mean[0, i], true_mean[0, i + 1], c='g',
                            marker='x', s=100, label='Initial state')
                plt.legend()
                plt.xlabel(r'$z_{}$'.format(i + 1))
                plt.ylabel(r'$z_{}$'.format(i + 2))
                plt.savefig(os.path.join(subfolder, name), bbox_inches='tight')
                plt.close('all')
        except FileNotFoundError:
            print('Files ' + os.path.join(subfolder, str(key)) + 'closedloop '
                                                                 'not found')


if __name__ == '__main__':
    # Collect data from given folder
    folder = str(input('Input folder from which to replot:\n'))

    # # Redo plots of variables of form(time, value)
    # vars_time = ['Loop_9/closedloop_rollout_RMSE', 'Loop_9/rollout_RMSE',
    #              'Loop_9/Data_outside_GP/Estimation_RMSE', 'Loop_9/RMSE_time']
    # replot_vars_time(subfolder=folder, variables=vars_time)

    # Rollouts to replot, from subfolder of one experiment (n dims)
    rollout_plots = ['Loop_0/Rollouts_-1/Rollout_11/',
                     'Loop_9/Rollouts_9/Rollout_11/',
                     'Loop_0/Rollouts_0/Rollout_60/',
                     'Loop_9/Rollouts_9/Rollout_60/']
    replot_rollouts(subfolder=folder, plots=rollout_plots)

    # # Variables to box plot, from folder containing several experiments,
    # # of form (time, value)
    # vars_time = ['Loop_9/closedloop_rollout_RMSE', 'Loop_9/rollout_RMSE',
    #              'Loop_9/Data_outside_GP/Estimation_RMSE', 'Loop_9/RMSE_time']
    # box_plots_vars_time(folder=folder, variables=vars_time, avoid0=False)
