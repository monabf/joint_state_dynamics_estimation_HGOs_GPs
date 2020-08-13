import logging
import os

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from config import Config
from plotting_closedloop_rollouts import save_closedloop_rollout_variables, \
    plot_test_closedloop_rollout_data, plot_val_closedloop_rollout_data
from plotting_functions import model_evaluation, run_rollouts
from plotting_kalman_rollouts import save_kalman_rollout_variables, \
    plot_val_kalman_rollout_data, plot_test_kalman_rollout_data
from plotting_rollouts import plot_val_rollout_data, plot_test_rollout_data, \
    save_rollout_variables
from simple_GP_dyn import Simple_GP_Dyn
from utils import remove_outlier, reshape_pt1

sb.set_style('whitegrid')


# Class to learn simple dynamics GP from several experimental datasets (hence
# usually ground truth is only approximated)
# Inherits from Simple_GP_Dyn, basically the same but ground_truth_approx =
# True by default and model evaluation tools (evaluation grid, rollouts,
# GP plot) are chosen close to training data and updated at each new learning
# loop to incorporate new training data


class GP_Dyn_Several_Exp_Datasets(Simple_GP_Dyn):

    def __init__(self, X, U, Y, config: Config):
        super().__init__(X=X, U=U, Y=Y, config=config, ground_truth_approx=True)

        if self.nb_rollouts > 1 and self.ground_truth_approx:
            logging.warning('No use in having a high number of rollouts when '
                            'no ground truth is available, since the real '
                            'model evaluation is obtained by predicting on a '
                            'test set, not on computing rollout RMSE or any '
                            'other type of metric using the ground truth!')
        self.test_RMSE = np.zeros((0, 2))
        self.test_SRMSE = np.zeros((0, 2))
        self.test_log_AL = np.zeros((0, 2))
        self.test_stand_log_AL = np.zeros((0, 2))
        self.val_RMSE = np.zeros((0, 2))
        self.val_SRMSE = np.zeros((0, 2))
        self.val_log_AL = np.zeros((0, 2))
        self.val_stand_log_AL = np.zeros((0, 2))
        self.test_rollout_RMSE = np.zeros((0, 2))
        self.test_rollout_SRMSE = np.zeros((0, 2))
        self.test_rollout_log_AL = np.zeros((0, 2))
        self.test_rollout_stand_log_AL = np.zeros((0, 2))
        self.val_rollout_RMSE = np.zeros((0, 2))
        self.val_rollout_SRMSE = np.zeros((0, 2))
        self.val_rollout_log_AL = np.zeros((0, 2))
        self.val_rollout_stand_log_AL = np.zeros((0, 2))
        self.observer_test_RMSE = np.zeros((0, 2))
        self.observer_test_SRMSE = np.zeros((0, 2))
        self.observer_val_RMSE = np.zeros((0, 2))
        self.observer_val_SRMSE = np.zeros((0, 2))
        self.output_test_RMSE = np.zeros((0, 2))
        self.output_test_SRMSE = np.zeros((0, 2))
        self.output_val_RMSE = np.zeros((0, 2))
        self.output_val_SRMSE = np.zeros((0, 2))
        self.test_closedloop_rollout_RMSE = np.zeros((0, 2))
        self.test_closedloop_rollout_SRMSE = np.zeros((0, 2))
        self.test_closedloop_rollout_log_AL = np.zeros((0, 2))
        self.test_closedloop_rollout_stand_log_AL = np.zeros((0, 2))
        self.val_closedloop_rollout_RMSE = np.zeros((0, 2))
        self.val_closedloop_rollout_SRMSE = np.zeros((0, 2))
        self.val_closedloop_rollout_log_AL = np.zeros((0, 2))
        self.val_closedloop_rollout_stand_log_AL = np.zeros((0, 2))
        self.test_kalman_rollout_RMSE = np.zeros((0, 2))
        self.test_kalman_rollout_SRMSE = np.zeros((0, 2))
        self.test_kalman_rollout_log_AL = np.zeros((0, 2))
        self.test_kalman_rollout_stand_log_AL = np.zeros((0, 2))
        self.val_kalman_rollout_RMSE = np.zeros((0, 2))
        self.val_kalman_rollout_SRMSE = np.zeros((0, 2))
        self.val_kalman_rollout_log_AL = np.zeros((0, 2))
        self.val_kalman_rollout_stand_log_AL = np.zeros((0, 2))

        if self.__class__.__name__ == 'GP_Dyn_Several_Exp_Datasets':
            # Only do if constructor not called from inherited class
            if not self.existing_results_folder:
                self.results_folder = \
                    self.results_folder.replace('_pass', '_fold_crossval')
                self.results_folder = self.results_folder.replace('/Loop_0', '')
            else:
                self.results_folder = self.existing_results_folder
            self.results_folder = os.path.join(self.results_folder,
                                               'Crossval_Fold_' + str(
                                                   self.fold_nb))
            if self.save_inside_fold:
                self.results_folder = os.path.join(self.results_folder,
                                                   'Loop_0')
            os.makedirs(self.results_folder, exist_ok=False)
            self.validation_folder = os.path.join(self.results_folder,
                                                  'Validation')
            os.makedirs(self.validation_folder, exist_ok=True)
            self.test_folder = os.path.join(self.results_folder, 'Test')
            os.makedirs(self.test_folder, exist_ok=True)
            self.save_grid_variables(self.grid, self.grid_controls,
                                     self.true_predicted_grid,
                                     self.results_folder)
            save_rollout_variables(self.results_folder, self.nb_rollouts,
                                   self.rollout_list, step=self.step,
                                   ground_truth_approx=self.ground_truth_approx)
            # Save log in results folder
            # if not existing_results_folder:
            #     os.rename('../Figures/Logs/' + 'log' + str(sys.argv[1]) +
            #               '.log', os.path.join(self.results_folder, 'log' + str(
            #         sys.argv[1]) + '.log'))
            # self.save_log(self.results_folder)
            if self.verbose:
                logging.info(self.results_folder)

    def update_data(self, new_X=[], new_U=[], new_Y=[]):
        whole_X, whole_U, whole_Y = Simple_GP_Dyn.update_data(self, new_X=new_X,
                                                              new_U=new_U,
                                                              new_Y=new_Y)

        # Recreate and resave evaluation grid and list of rollouts with updated
        # training data
        old_X = self.X
        old_U = self.U
        old_Y = self.Y
        self.X = whole_X
        self.U = whole_U
        self.Y = whole_Y
        self.grid, self.grid_controls = self.create_grid(self.init_state,
                                                         self.init_control,
                                                         self.constrain_u,
                                                         self.grid_inf,
                                                         self.grid_sup)
        true_predicted_grid = \
            self.create_true_predicted_grid(self.grid, self.grid_controls)
        self.true_predicted_grid = \
            true_predicted_grid.reshape(-1, self.Y.shape[1])
        # Reject outliers from grid
        true_predicted_grid_df = pd.DataFrame(self.true_predicted_grid)
        grid_df = pd.DataFrame(self.grid)
        grid_controls_df = pd.DataFrame(self.grid_controls)
        mask = remove_outlier(true_predicted_grid_df)
        true_predicted_grid_df = true_predicted_grid_df[mask]
        grid_df = grid_df[mask]
        grid_controls_df = grid_controls_df[mask]
        self.true_predicted_grid = true_predicted_grid_df.values
        self.grid = grid_df.values
        self.grid_controls = grid_controls_df.values
        self.true_predicted_grid = \
            self.true_predicted_grid.reshape(-1, self.Y.shape[1])
        self.grid = self.grid.reshape(-1, self.X.shape[1])
        self.grid_controls = self.grid_controls.reshape(-1, self.U.shape[1])
        self.grid_variables = {'Evaluation_grid': self.grid,
                               'Grid_controls': np.array(self.grid_controls)}
        self.save_grid_variables(self.grid, self.grid_controls,
                                 self.true_predicted_grid, self.results_folder)
        self.rollout_list = self.create_rollout_list()
        save_rollout_variables(self.results_folder, self.nb_rollouts,
                               self.rollout_list, step=self.step - 1,
                               ground_truth_approx=self.ground_truth_approx)
        self.X = old_X
        self.U = old_U
        self.Y = old_Y
        del whole_X, whole_U, whole_Y

    def evaluate_test_rollouts(self, only_prior=False):
        if len(self.test_rollout_list) == 0:
            return 0

        save_rollout_variables(self.test_folder, len(self.test_rollout_list),
                               self.test_rollout_list, step=self.step - 1,
                               results=False,
                               ground_truth_approx=self.ground_truth_approx,
                               title='Test_rollouts')
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, np.array(self.test_rollout_list),
                         folder=self.test_folder, only_prior=only_prior)
        self.specs['test_rollout_RMSE'] = rollout_RMSE
        self.specs['test_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_rollout_log_AL'] = rollout_log_AL
        self.specs['test_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.test_rollout_RMSE = \
            np.concatenate((self.test_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.test_rollout_SRMSE = \
            np.concatenate((self.test_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.test_rollout_log_AL = \
            np.concatenate((self.test_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.test_rollout_stand_log_AL = \
            np.concatenate((self.test_rollout_stand_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['test_rollout_RMSE'] = self.test_rollout_RMSE
        self.variables['test_rollout_SRMSE'] = self.test_rollout_SRMSE
        self.variables['test_rollout_log_AL'] = self.test_rollout_log_AL
        self.variables['test_rollout_stand_log_AL'] = \
            self.test_rollout_stand_log_AL
        plot_test_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = np.concatenate((self.test_rollout_list,
                                                     rollout_list), axis=1)
        save_rollout_variables(
            self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1, results=True,
            ground_truth_approx=self.ground_truth_approx, title='Test_rollouts')

    def evaluate_test_kalman_rollouts(self, observer, observe_data,
                                      discrete_observer,
                                      no_GP_in_observer=False,
                                      only_prior=False):
        if len(self.test_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, np.array(self.test_rollout_list), folder=self.test_folder,
            observer=observer, observe_data=observe_data,
            discrete_observer=discrete_observer, kalman=True,
            no_GP_in_observer=no_GP_in_observer, only_prior=only_prior)
        self.specs['test_kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['test_kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['test_kalman_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.test_kalman_rollout_RMSE = \
            np.concatenate((self.test_kalman_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.test_kalman_rollout_SRMSE = \
            np.concatenate((self.test_kalman_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.test_kalman_rollout_log_AL = \
            np.concatenate((self.test_kalman_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.test_kalman_rollout_stand_log_AL = np.concatenate((
            self.test_kalman_rollout_stand_log_AL, reshape_pt1(np.array(
                [self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['test_kalman_rollout_RMSE'] = \
            self.test_kalman_rollout_RMSE
        self.variables['test_kalman_rollout_SRMSE'] = \
            self.test_kalman_rollout_SRMSE
        self.variables['test_kalman_rollout_log_AL'] = \
            self.test_kalman_rollout_log_AL
        self.variables['test_kalman_rollout_stand_log_AL'] = \
            self.test_kalman_rollout_stand_log_AL
        plot_test_kalman_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = np.concatenate((self.test_rollout_list,
                                                     rollout_list), axis=1)
        save_kalman_rollout_variables(
            self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx, title='Test_rollouts')

    def evaluate_test_closedloop_rollouts(self, observer, observe_data,
                                          no_GP_in_observer=False):
        if len(self.test_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, np.array(self.test_rollout_list), folder=self.test_folder,
            observer=observer, observe_data=observe_data, closedloop=True,
            no_GP_in_observer=no_GP_in_observer)
        self.specs['test_closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['test_closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['test_closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['test_closedloop_rollout_stand_log_AL'] = \
            rollout_stand_log_AL
        self.test_closedloop_rollout_RMSE = \
            np.concatenate((self.test_closedloop_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.test_closedloop_rollout_SRMSE = \
            np.concatenate((self.test_closedloop_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.test_closedloop_rollout_log_AL = \
            np.concatenate((self.test_closedloop_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.test_closedloop_rollout_stand_log_AL = np.concatenate((
            self.test_closedloop_rollout_stand_log_AL, reshape_pt1(np.array(
                [self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['test_closedloop_rollout_RMSE'] = \
            self.test_closedloop_rollout_RMSE
        self.variables['test_closedloop_rollout_SRMSE'] = \
            self.test_closedloop_rollout_SRMSE
        self.variables['test_closedloop_rollout_log_AL'] = \
            self.test_closedloop_rollout_log_AL
        self.variables['test_closedloop_rollout_stand_log_AL'] = \
            self.test_closedloop_rollout_stand_log_AL
        plot_test_closedloop_rollout_data(self, folder=self.test_folder)
        complete_test_rollout_list = np.concatenate((self.test_rollout_list,
                                                     rollout_list), axis=1)
        save_closedloop_rollout_variables(
            self.test_folder, len(complete_test_rollout_list),
            complete_test_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx, title='Test_rollouts')

    def evaluate_val_rollouts(self, only_prior=False):
        if len(self.val_rollout_list) == 0:
            return 0

        save_rollout_variables(self.validation_folder,
                               len(self.val_rollout_list),
                               self.val_rollout_list, step=self.step - 1,
                               results=False,
                               ground_truth_approx=self.ground_truth_approx,
                               title='Val_rollouts')
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, np.array(self.val_rollout_list),
            folder=self.validation_folder, only_prior=only_prior)
        self.specs['val_rollout_RMSE'] = rollout_RMSE
        self.specs['val_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_rollout_log_AL'] = rollout_log_AL
        self.specs['val_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.val_rollout_RMSE = \
            np.concatenate((self.val_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.val_rollout_SRMSE = \
            np.concatenate((self.val_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.val_rollout_log_AL = \
            np.concatenate((self.val_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.val_rollout_stand_log_AL = \
            np.concatenate((self.val_rollout_stand_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['val_rollout_RMSE'] = self.val_rollout_RMSE
        self.variables['val_rollout_SRMSE'] = self.val_rollout_SRMSE
        self.variables['val_rollout_log_AL'] = self.val_rollout_log_AL
        self.variables['val_rollout_stand_log_AL'] = \
            self.val_rollout_stand_log_AL
        plot_val_rollout_data(self, folder=self.validation_folder)
        complete_val_rollout_list = np.concatenate((self.val_rollout_list,
                                                    rollout_list), axis=1)
        save_rollout_variables(
            self.validation_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1, results=True,
            ground_truth_approx=self.ground_truth_approx, title='Val_rollouts')

    def evaluate_val_kalman_rollouts(self, observer, observe_data,
                                     discrete_observer,
                                     no_GP_in_observer=False,
                                     only_prior=False):
        if len(self.val_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, np.array(self.val_rollout_list),
            folder=self.validation_folder, observer=observer,
            observe_data=observe_data, discrete_observer=discrete_observer,
            kalman=True, no_GP_in_observer=no_GP_in_observer,
            only_prior=only_prior)
        self.specs['val_kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['val_kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['val_kalman_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.val_kalman_rollout_RMSE = \
            np.concatenate((self.val_kalman_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.val_kalman_rollout_SRMSE = \
            np.concatenate((self.val_kalman_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.val_kalman_rollout_log_AL = \
            np.concatenate((self.val_kalman_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.val_kalman_rollout_stand_log_AL = np.concatenate((
            self.val_kalman_rollout_stand_log_AL, reshape_pt1(np.array(
                [self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['val_kalman_rollout_RMSE'] = \
            self.val_kalman_rollout_RMSE
        self.variables['val_kalman_rollout_SRMSE'] = \
            self.val_kalman_rollout_SRMSE
        self.variables['val_kalman_rollout_log_AL'] = \
            self.val_kalman_rollout_log_AL
        self.variables['val_kalman_rollout_stand_log_AL'] = \
            self.val_kalman_rollout_stand_log_AL
        plot_val_kalman_rollout_data(self, folder=self.validation_folder)
        complete_val_rollout_list = np.concatenate((self.val_rollout_list,
                                                    rollout_list), axis=1)
        save_kalman_rollout_variables(
            self.validation_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx, title='Val_rollouts')

    def evaluate_val_closedloop_rollouts(self, observer, observe_data,
                                         no_GP_in_observer=False):
        if len(self.val_rollout_list) == 0:
            return 0

        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = run_rollouts(
            self, np.array(self.val_rollout_list),
            folder=self.validation_folder, observer=observer,
            observe_data=observe_data, closedloop=True,
            no_GP_in_observer=no_GP_in_observer)
        self.specs['val_closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['val_closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['val_closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['val_closedloop_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.val_closedloop_rollout_RMSE = \
            np.concatenate((self.val_closedloop_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.val_closedloop_rollout_SRMSE = \
            np.concatenate((self.val_closedloop_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.val_closedloop_rollout_log_AL = \
            np.concatenate((self.val_closedloop_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.val_closedloop_rollout_stand_log_AL = np.concatenate((
            self.val_closedloop_rollout_stand_log_AL, reshape_pt1(np.array(
                [self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['val_closedloop_rollout_RMSE'] = \
            self.val_closedloop_rollout_RMSE
        self.variables['val_closedloop_rollout_SRMSE'] = \
            self.val_closedloop_rollout_SRMSE
        self.variables['val_closedloop_rollout_log_AL'] = \
            self.val_closedloop_rollout_log_AL
        self.variables['val_closedloop_rollout_stand_log_AL'] = \
            self.val_closedloop_rollout_stand_log_AL
        plot_val_closedloop_rollout_data(self, folder=self.validation_folder)
        complete_val_rollout_list = np.concatenate((self.val_rollout_list,
                                                    rollout_list), axis=1)
        save_closedloop_rollout_variables(
            self.validation_folder, len(complete_val_rollout_list),
            complete_val_rollout_list, step=self.step - 1,
            ground_truth_approx=self.ground_truth_approx, title='Val_rollouts')

    def test(self, X_test, U_test, Y_test, cut_idx=[]):
        # Save test data
        if not X_test.any():
            self.test_rollout_list = []
            self.nb_test_rollouts = 0
            return 0

        self.test_folder = os.path.join(self.results_folder, 'Test')
        os.makedirs(self.test_folder, exist_ok=True)
        name = 'X_test'
        for i in range(X_test.shape[1]):
            plt.plot(X_test[:, i], label='Test input')
            plt.title('Input data used for testing')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Input')
            plt.savefig(os.path.join(self.test_folder, name + str(i) + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(X_test)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)
        name = 'U_test'
        for i in range(U_test.shape[1]):
            plt.plot(U_test[:, i], label='Test control')
            plt.title('Control data used for testing')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Control')
            plt.savefig(os.path.join(self.test_folder, name + str(i) + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(U_test)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)
        name = 'Y_test'
        for i in range(Y_test.shape[1]):
            plt.plot(Y_test[:, i], label='Test output')
            plt.title('Output data used for testing')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Output')
            plt.savefig(os.path.join(self.test_folder, name + str(i) + '.pdf'),
                        bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(Y_test)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)

        # Compute RMSE and log_AL over test data
        if 'Michelangelo' in self.system:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(grid=X_test, grid_controls=U_test,
                                          true_predicted_grid=Y_test,
                                          use_euler='Michelangelo')
        elif ('justvelocity' in self.system) and not self.continuous_model:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(grid=X_test, grid_controls=U_test,
                                          true_predicted_grid=Y_test,
                                          use_euler='discrete_justvelocity')
        elif ('justvelocity' in self.system) and self.continuous_model:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(grid=X_test, grid_controls=U_test,
                                          true_predicted_grid=Y_test,
                                          use_euler='continuous_justvelocity')
        else:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(grid=X_test, grid_controls=U_test,
                                          true_predicted_grid=Y_test,
                                          use_euler=None)
        self.test_RMSE = np.concatenate((self.test_RMSE, reshape_pt1(
            np.array([self.sample_idx, RMSE]))), axis=0)
        self.test_SRMSE = np.concatenate((self.test_SRMSE, reshape_pt1(
            np.array([self.sample_idx, SRMSE]))), axis=0)
        self.test_log_AL = np.concatenate((self.test_log_AL, reshape_pt1(
            np.array([self.sample_idx, log_likelihood]))), axis=0)
        self.test_stand_log_AL = np.concatenate((
            self.test_stand_log_AL, reshape_pt1(np.array(
                [self.sample_idx, stand_log_likelihood]))), axis=0)

        # Save plot and csv files of test_RMSE and test_log_AL
        name = 'Test_RMSE'
        plt.plot(self.test_RMSE[:, 0] - 1, self.test_RMSE[:, 1], 'c',
                 label='RMSE')
        plt.title('RMSE between model and true dynamics over test data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('RMSE')
        plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.test_RMSE)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)

        name = 'Test_SRMSE'
        plt.plot(self.test_SRMSE[:, 0] - 1, self.test_SRMSE[:, 1], 'c',
                 label='SRMSE')
        plt.title('Standardized RMSE between model and true dynamics over test '
                  'data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('SRMSE')
        plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.test_SRMSE)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)

        name = 'Test_average_log_likelihood'
        plt.plot(self.test_log_AL[:, 0], self.test_log_AL[:, 1], 'c',
                 label='log_AL')
        plt.title('Average log likelihood between model and true dynamics '
                  'over test data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Average log likelihood')
        plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.test_log_AL)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)

        name = 'Test_standardized_average_log_likelihood'
        plt.plot(self.test_stand_log_AL[:, 0], self.test_stand_log_AL[:, 1],
                 'c', label='stand_log_AL')
        plt.title('Standardized average log likelihood between model and true '
                  'dynamics over test data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Standardized average log likelihood')
        plt.savefig(os.path.join(self.test_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.test_stand_log_AL)
        file.to_csv(os.path.join(self.test_folder, name + '.csv'),
                    header=False)

        # Plot model evaluation scatter plot over the test data
        model_evaluation(Evaluation_grid=X_test, Grid_controls=U_test,
                         Predicted_grid=predicted_grid,
                         True_predicted_grid=Y_test,
                         folder=self.test_folder,
                         ground_truth_approx=self.ground_truth_approx,
                         title='Test_model_evaluation', verbose=False)
        self.save_grid_variables(X_test, U_test, Y_test,
                                 results_folder=self.test_folder)
        filename = 'Predicted_Ytest' + str(self.step) + '.csv'
        file = pd.DataFrame(predicted_grid)
        file.to_csv(os.path.join(self.test_folder, filename), header=False)

        # Rollouts over the test data: either random subsets of test data,
        # number proportional to length of test data, or full test scenario
        # for each rollout
        if (self.fold_nb == 0) and (self.step == 1):
            test_rollout_list = []
            for i in range(len(cut_idx)):
                start_idx = cut_idx[i]  # start of test traj
                if i < len(cut_idx) - 1:
                    end_idx = cut_idx[i + 1]  # end of test traj (excluded)
                else:
                    end_idx = len(Y_test)
                if self.full_rollouts:
                    init_state = reshape_pt1(X_test[start_idx])
                    true_mean = np.concatenate((
                        init_state, reshape_pt1(Y_test[start_idx:end_idx])),
                        axis=0)
                    control_traj = reshape_pt1(U_test[start_idx:end_idx])
                    test_rollout_list.append(
                        [init_state, control_traj, true_mean])
                else:
                    nb_local_rollouts = int(
                        np.floor((end_idx - start_idx) / self.rollout_length))
                    for j in range(nb_local_rollouts):
                        random_start_idx = np.random.randint(
                            start_idx, end_idx - self.rollout_length)
                        random_end_idx = random_start_idx + self.rollout_length
                        init_state = reshape_pt1(X_test[random_start_idx])
                        # true_mean = reshape_pt1(
                        #     X_test[random_start_idx:random_end_idx + 1])
                        true_mean = np.concatenate((
                            init_state, reshape_pt1(
                                Y_test[random_start_idx:random_end_idx])),
                            axis=0)
                        control_traj = reshape_pt1(
                            U_test[random_start_idx:random_end_idx])
                        test_rollout_list.append(
                            [init_state, control_traj, true_mean])
            self.test_rollout_list = np.array(test_rollout_list)
            self.nb_test_rollouts = len(self.test_rollout_list)
        elif (self.fold_nb > 0) and (self.step == 1):
            previous_test_folder = self.test_folder.replace(
                'Crossval_Fold_' + str(self.fold_nb), 'Crossval_Fold_0')
            path, dirs, files = next(os.walk(os.path.join(
                previous_test_folder, 'Test_rollouts0')))
            self.nb_test_rollouts = len(dirs)
            self.test_rollout_list = self.read_rollout_list(
                previous_test_folder, self.nb_test_rollouts, step=self.step - 1,
                folder_title='Test_rollouts')
        self.evaluate_test_rollouts()

    def validate(self, X_val, U_val, Y_val, cut_idx=[]):
        # Save validation data
        if not X_val.any():
            self.val_rollout_list = []
            self.nb_val_rollouts = 0
            return 0

        self.validation_folder = os.path.join(self.results_folder, 'Validation')
        os.makedirs(self.validation_folder, exist_ok=True)
        name = 'X_val'
        for i in range(X_val.shape[1]):
            plt.plot(X_val[:, i], label='Validation input')
            plt.title('Input data used for validation')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Input')
            plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                     '.pdf'), bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(X_val)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)
        name = 'U_val'
        for i in range(U_val.shape[1]):
            plt.plot(U_val[:, i], label='Validation control')
            plt.title('Control data used for validation')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Control')
            plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                     '.pdf'), bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(U_val)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)
        name = 'Y_val'
        for i in range(Y_val.shape[1]):
            plt.plot(Y_val[:, i], label='Validation output')
            plt.title('Output data used for validation')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Output')
            plt.savefig(os.path.join(self.validation_folder, name + str(i) +
                                     '.pdf'), bbox_inches='tight')
            plt.close('all')
        file = pd.DataFrame(Y_val)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)

        # Compute RMSE and log_AL over val data
        if 'Michelangelo' in self.system:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(
                    grid=X_val, grid_controls=U_val, true_predicted_grid=Y_val,
                    use_euler='Michelangelo')
        elif ('justvelocity' in self.system) and not self.continuous_model:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(
                    grid=X_val, grid_controls=U_val, true_predicted_grid=Y_val,
                    use_euler='discrete_justvelocity')
        elif ('justvelocity' in self.system) and self.continuous_model:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(
                    grid=X_val, grid_controls=U_val, true_predicted_grid=Y_val,
                    use_euler='continuous_justvelocity')
        else:
            RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
            grid_controls, log_likelihood, stand_log_likelihood = \
                self.compute_l2error_grid(
                    grid=X_val, grid_controls=U_val, true_predicted_grid=Y_val,
                    use_euler=None)
        self.val_RMSE = np.concatenate((self.val_RMSE, reshape_pt1(
            np.array([self.sample_idx, RMSE]))), axis=0)
        self.val_SRMSE = np.concatenate((self.val_SRMSE, reshape_pt1(
            np.array([self.sample_idx, SRMSE]))), axis=0)
        self.val_log_AL = np.concatenate((self.val_log_AL, reshape_pt1(
            np.array([self.sample_idx, log_likelihood]))), axis=0)
        self.val_stand_log_AL = \
            np.concatenate((self.val_stand_log_AL, reshape_pt1(np.array([
                self.sample_idx, stand_log_likelihood]))), axis=0)

        # Save plot and csv files of val_RMSE and val_log_AL
        name = 'Validation_RMSE'
        plt.plot(self.val_RMSE[:, 0] - 1, self.val_RMSE[:, 1], 'c',
                 label='RMSE')
        plt.title('RMSE between model and true dynamics over validation data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('RMSE')
        plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.val_RMSE)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)

        name = 'Validation_SRMSE'
        plt.plot(self.val_SRMSE[:, 0] - 1, self.val_SRMSE[:, 1], 'c',
                 label='SRMSE')
        plt.title('Standardized RMSE between model and true dynamics over '
                  'validation data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('SRMSE')
        plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.val_SRMSE)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)

        name = 'Validation_average_log_likelihood'
        plt.plot(self.val_log_AL[:, 0], self.val_log_AL[:, 1], 'c',
                 label='log_AL')
        plt.title('Average log likelihood between model and true dynamics '
                  'over validation data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Average log likelihood')
        plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.val_log_AL)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)

        name = 'Validation_standardized_average_log_likelihood'
        plt.plot(self.val_stand_log_AL[:, 0], self.val_stand_log_AL[:, 1], 'c',
                 label='stand_log_AL')
        plt.title('Standardized average log likelihood between model and true '
                  'dynamics over validation data')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Standardized average log likelihood')
        plt.savefig(os.path.join(self.validation_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')
        file = pd.DataFrame(self.val_stand_log_AL)
        file.to_csv(os.path.join(self.validation_folder, name + '.csv'),
                    header=False)

        # Plot model evaluation scatter plot over the val data
        model_evaluation(Evaluation_grid=X_val, Grid_controls=U_val,
                         Predicted_grid=predicted_grid,
                         True_predicted_grid=Y_val,
                         folder=self.validation_folder,
                         ground_truth_approx=self.ground_truth_approx,
                         title='Val_model_evaluation', verbose=False)
        self.save_grid_variables(X_val, U_val, Y_val,
                                 results_folder=self.validation_folder)
        filename = 'Predicted_Yval' + str(self.step) + '.csv'
        file = pd.DataFrame(predicted_grid)
        file.to_csv(os.path.join(self.validation_folder, filename),
                    header=False)

        # Rollouts over the val data: random subsets of val data, number
        # proportional to length of val data, or full validation scenario for
        # each rollout
        if self.step == 1:
            val_rollout_list = []
            for i in range(len(cut_idx)):
                start_idx = cut_idx[i]  # start of val traj
                if i < len(cut_idx) - 1:
                    end_idx = cut_idx[i + 1]  # end of val traj (excluded)
                else:
                    end_idx = len(Y_val)
                if self.full_rollouts:
                    init_state = reshape_pt1(X_val[start_idx])
                    true_mean = reshape_pt1(X_val[start_idx:end_idx + 1])
                    control_traj = reshape_pt1(U_val[start_idx:end_idx])
                    val_rollout_list.append(
                        [init_state, control_traj, true_mean])
                else:
                    nb_local_rollouts = int(
                        np.floor((end_idx - start_idx) / self.rollout_length))
                    for j in range(nb_local_rollouts):
                        random_start_idx = np.random.randint(
                            start_idx, end_idx - self.rollout_length)
                        random_end_idx = random_start_idx + self.rollout_length
                        init_state = reshape_pt1(X_val[random_start_idx])
                        true_mean = reshape_pt1(
                            X_val[random_start_idx:random_end_idx + 1])
                        control_traj = reshape_pt1(
                            U_val[random_start_idx:random_end_idx])
                        val_rollout_list.append(
                            [init_state, control_traj, true_mean])
            self.val_rollout_list = np.array(val_rollout_list)
            self.nb_val_rollouts = len(self.val_rollout_list)
        self.evaluate_val_rollouts()
