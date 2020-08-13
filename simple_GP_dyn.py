import logging
import os
import sys
import time

import GPy
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from multioutput_GPy_wrapper import MultiOutput_GPy_Wrapper
from sklearn import preprocessing

from config import Config
from controllers import sin_controller_1D, sin_controller_02D, null_controller
from plotting_closedloop_rollouts import plot_closedloop_rollout_data, \
    save_closedloop_rollout_variables
from plotting_functions import model_evaluation, save_GP_data, plot_GP, \
    run_rollouts
from plotting_kalman_rollouts import save_kalman_rollout_variables, \
    plot_kalman_rollout_data
from plotting_rollouts import plot_rollout_data, save_rollout_variables
from utils import reshape_pt1, remove_outlier, RMS, \
    log_multivariate_normal_likelihood, reshape_pt1_tonormal, reshape_dim1

sb.set_style('whitegrid')


# Class to learn simple dynamics GP from mappings (X,U)=(x_t, u_t) -> Y=(y_t)
# or (x_t+1), that already contain noise chosen by the user. Takes care of
# evaluating/saving GP and other utilities.
# For free system where U=0 uniformly: make U a matrix of zeros,
# control ignored for learning


class Simple_GP_Dyn:

    def __getattr__(self, item):
        return self.config.__getattr__(item)

    def __init__(self, X, U, Y, config: Config, ground_truth_approx=False):
        assert len(X) == len(U) == len(Y), 'X, U and Y must have the same ' \
                                           'length'
        self.config = config
        self.specs = self.config.params
        self.X = reshape_pt1(X)
        self.U = reshape_pt1(U)
        self.Y = reshape_pt1(Y)

        if self.dt > 0.1:
            logging.warning('Time step is larger than 0.1s! This might be too '
                            'much, most of all for all small operations that '
                            'rely on Euler discretization to obtain continuous '
                            'solutions from discrete GP models, such as '
                            'rollouts, continuous observers...')
        if ((X.shape[1] != Y.shape[1]) and (self.nb_rollouts > 0)) and not \
                (('Michelangelo' in self.system) or (
                        'justvelocity' in self.system)):
            raise ValueError('The rollout function is only available for '
                             'dynamics GPs learning x_t -> x_t+1, x_t -> '
                             'velocity_t+1, x_t -> velocitydot_t, '
                             'or in particular cases where it has been '
                             'precoded. To learn another type of GP '
                             'x_t -> y_t, set the number of rollouts to zero.')
        if self.hyperparam_optim:
            assert any(self.hyperparam_optim == k for k in (
                'fixed_start', 'fixed_hyperparameters')), \
                'Only possible options for hyperparam_optim are None, ' \
                'fixed_start, and fixed_hyperparameters.'
        if self.sparse:
            assert isinstance(self.sparse, dict), \
                'The sparse parameter must be a dictionary containing the ' \
                'method (VarDTC or FITC) and the number of inducing inputs ' \
                'nb_inducing_inputs, which are otherwise set to default.'
        if not self.grid_inf:
            logging.warning('No grid was predefined by the user for one step '
                            'ahead model evaluation and rollouts, so using '
                            'min and max of state data.')
            self.grid_inf = np.min(self.X, axis=0)
            self.grid_sup = np.max(self.X, axis=0)

        self.step = 0
        self.sample_idx = 0
        if ground_truth_approx:
            # Data rollouts cannot be longer than data
            self.rollout_length = int(
                np.min([self.rollout_length, len(self.X) - 1]))
        self.prior_kwargs = self.config.prior_kwargs
        self.ground_truth_approx = ground_truth_approx
        if self.ground_truth_approx:
            logging.warning('True dynamics are approximated from data or '
                            'from a simplified model: there is actually no '
                            'ground truth, the true dynamics are only used as '
                            'a comparison to the GP model! Hence, model '
                            'evaluation tools such as GP_plot, rollouts or '
                            'model_evaluation are only indicative; true '
                            'evaluation of the model can only be obtained by '
                            'predicting on a test set and comparing to the '
                            'true data.')
        self.hyperparams = np.array([[]])
        # Metrics are important to monitor learning! See
        # https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
        # and GPML, Rasmussen, Section 2.5
        self.RMSE_time = np.zeros((0, 2))
        self.SRMSE_time = np.zeros((0, 2))
        self.log_likelihood_time = np.zeros((0, 2))
        self.stand_log_likelihood_time = np.zeros((0, 2))
        self.stop = False
        self.current_time = time.time() - time.time()
        self.time = reshape_pt1(np.array([[0, 0]]))
        self.observer_RMSE = np.zeros((0, 2))
        self.observer_SRMSE = np.zeros((0, 2))
        self.output_RMSE = np.zeros((0, 2))
        self.output_SRMSE = np.zeros((0, 2))
        self.rollout_RMSE = np.zeros((0, 2))
        self.rollout_SRMSE = np.zeros((0, 2))
        self.rollout_log_AL = np.zeros((0, 2))
        self.rollout_stand_log_AL = np.zeros((0, 2))
        self.kalman_rollout_RMSE = np.zeros((0, 2))
        self.kalman_rollout_SRMSE = np.zeros((0, 2))
        self.kalman_rollout_log_AL = np.zeros((0, 2))
        self.kalman_rollout_stand_log_AL = np.zeros((0, 2))
        self.closedloop_rollout_RMSE = np.zeros((0, 2))
        self.closedloop_rollout_SRMSE = np.zeros((0, 2))
        self.closedloop_rollout_log_AL = np.zeros((0, 2))
        self.closedloop_rollout_stand_log_AL = np.zeros((0, 2))

        # Create grid of (x_t, u_t) to evaluate the GP quality (true dynamics
        # needed to compare true x_t+1 to predicted)
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
        self.rollout_list = self.create_rollout_list()

        self.variables = {'X': self.X, 'Y': self.Y,
                          'Computation_time': np.array(self.time)}
        self.variables['RMSE_time'] = np.array(self.RMSE_time)
        self.variables['SRMSE_time'] = np.array(self.SRMSE_time)
        self.variables['log_AL_time'] = np.array(self.log_likelihood_time)
        self.variables['stand_log_AL_time'] = np.array(
            self.stand_log_likelihood_time)
        self.grid_variables = {'Evaluation_grid': self.grid,
                               'Grid_controls': np.array(self.grid_controls)}

        # Create unique results folder
        params = str(np.random.uniform()) + '_' + str(
            self.nb_samples) + 'samples_' + 't' + str(
            self.specs.get('t0')) + '-' + str(self.specs.get('tf')) + '_' + str(
            self.true_meas_noise_var) + 'meas_' + str(
            self.process_noise_var) + 'process'
        if self.restart_on_loop:
            params = 'Restarts_' + params
        if self.kernel.name != 'rbf':
            params += '_kernel_' + str(self.kernel.name)
        if not self.constrain_u:
            params += '_unclipped'
        else:
            params += '_uclip' + str(np.max(self.constrain_u))
        if self.nb_loops > 1:
            self.results_folder = os.path.join('../Figures', str(self.system),
                                               str(self.nb_loops) + '_pass',
                                               str(self.nb_rollouts) +
                                               '_rollouts', params, 'Loop_0')
        else:
            self.results_folder = os.path.join(
                '../Figures', str(self.system), 'Single_pass',
                str(self.nb_rollouts) + '_rollouts', params)
        if self.__class__.__name__ == 'Simple_GP_Dyn':
            # Only do if constructor not called from inherited class
            os.makedirs(self.results_folder, exist_ok=False)
            self.save_grid_variables(self.grid, self.grid_controls,
                                     self.true_predicted_grid,
                                     self.results_folder)
            save_rollout_variables(self.results_folder, self.nb_rollouts,
                                   self.rollout_list, step=self.step,
                                   ground_truth_approx=self.ground_truth_approx)
            # Save log in results folder
            os.rename('../Figures/Logs/' + 'log' + str(sys.argv[1]) + '.log',
                      os.path.join(self.results_folder,
                                   'log' + str(sys.argv[1]) + '.log'))
            self.save_log(self.results_folder)
            if self.verbose:
                logging.info(self.results_folder)

    def learn(self, new_X=[], new_U=[], new_Y=[]):
        self.step += 1
        if self.verbose:
            logging.info('Update GP for the ' + str(self.step) + 'th time')
        logging.getLogger("paramz").setLevel(logging.ERROR)

        # Update data, model and hyperparameters
        self.update_data(new_X=new_X, new_U=new_U, new_Y=new_Y)
        start = time.time()
        self.update_model()
        # Record computation time for learning (not for other tasks)
        self.current_time += time.time() - start
        self.time = np.concatenate((self.time, reshape_pt1(
            np.array([self.sample_idx, self.current_time]))), axis=0)
        if self.monitor_experiment:
            self.evaluate_model()
        return self.model

    def create_GP_model(self):
        # Create GP model
        if self.multioutput_GP:
            self.model = MultiOutput_GPy_Wrapper(self.GP_X, self.GP_Y,
                                                 kernel=self.kernel,
                                                 likelihood=GPy.likelihoods.Gaussian(
                                                     variance=self.meas_noise_var),
                                                 inference_method=GPy.inference.latent_function_inference.ExactGaussianInference())
            for k in range(self.model.nb_output_dims):
                self.model.models[k].preferred_optimizer = self.GP_optim_method
                for i in range(len(self.model.models[k].kern.parameters)):
                    self.model.models[k].kern.parameters[i].constrain_bounded(
                        1e-3, 200.)
        elif self.sparse:
            if self.sparse.get('nb_inducing_inputs'):
                self.nb_inducing_inputs = np.min([self.sparse.get(
                    'nb_inducing_inputs'), len(self.GP_X)])
            else:
                self.nb_inducing_inputs = int(np.floor(len(self.GP_X) / 10))
                self.sparse.update({'default_nb_inducing_inputs': 'len(X)/10'})
            random_idx = np.random.choice(len(self.GP_X),
                                          self.nb_inducing_inputs)
            if self.nb_inducing_inputs == 1:
                Z = reshape_pt1(self.GP_X[random_idx])
            else:
                Z = reshape_dim1(self.GP_X[random_idx])
            if self.nb_inducing_inputs >= len(self.GP_X):
                logging.warning('More inducing points than data, using '
                                'regular GP instead of sparse approximation.')
                self.model = GPy.core.gp.GP(self.GP_X, self.GP_Y,
                                            kernel=self.kernel,
                                            likelihood=GPy.likelihoods.Gaussian(
                                                variance=self.meas_noise_var),
                                            inference_method=GPy.inference.latent_function_inference.ExactGaussianInference())
            elif self.sparse.get('method') == 'FITC':
                self.model = GPy.core.SparseGP(self.GP_X, self.GP_Y, Z,
                                               kernel=self.kernel,
                                               likelihood=GPy.likelihoods.Gaussian(
                                                   variance=self.meas_noise_var),
                                               inference_method=GPy.inference.latent_function_inference.FITC()
                                               )
            elif (self.sparse.get('method') == 'VarDTC') or \
                    not (self.sparse.get('method')):
                self.model = GPy.core.SparseGP(self.GP_X, self.GP_Y, Z,
                                               kernel=self.kernel,
                                               likelihood=GPy.likelihoods.Gaussian(
                                                   variance=self.meas_noise_var),
                                               inference_method=GPy.inference.latent_function_inference.VarDTC()
                                               )
            else:
                logging.error('This sparsification method is not implemented, '
                              'please use VarDTC.')
            self.model.preferred_optimizer = self.GP_optim_method
            for i in range(len(self.model.kern.parameters)):
                self.model.kern.parameters[i].constrain_bounded(1e-3, 200.)
        else:
            self.model = GPy.core.gp.GP(self.GP_X, self.GP_Y,
                                        kernel=self.kernel,
                                        likelihood=GPy.likelihoods.Gaussian(
                                            variance=self.meas_noise_var),
                                        inference_method=GPy.inference.latent_function_inference.ExactGaussianInference())
            self.model.preferred_optimizer = self.GP_optim_method
            for i in range(len(self.model.kern.parameters)):
                self.model.kern.parameters[i].constrain_bounded(1e-3, 200.)
        if self.hyperparam_optim and self.step == 1:
            # Fix the starting point for hyperparameter optimization to the
            # parameters at initialization
            if self.hyperparam_optim == 'fixed_start':
                self.start_optim_fixed = self.model.optimizer_array.copy()

    def update_data(self, new_X=[], new_U=[], new_Y=[]):
        # Update GP with standardized, noisy observation data
        if self.step > 1 and self.memory_saving:
            # At each GP update, write only new data to file and read whole
            self.save_intermediate(memory_saving=True)
            X = pd.read_csv(
                os.path.join(self.results_folder, 'X.csv'), sep=',',
                header=None)
            whole_X = X.drop(X.columns[0], axis=1).values
            U = pd.read_csv(
                os.path.join(self.results_folder, 'U.csv'), sep=',',
                header=None)
            whole_U = U.drop(U.columns[0], axis=1).values
            Y = pd.read_csv(
                os.path.join(self.results_folder, 'Y.csv'), sep=',',
                header=None)
            whole_Y = Y.drop(U.columns[0], axis=1).values
        elif self.step <= 1 and self.memory_saving:
            self.save_intermediate(memory_saving=False)
            whole_X = self.X
            whole_U = self.U
            whole_Y = self.Y
        else:
            whole_X = self.X
            whole_U = self.U
            whole_Y = self.Y

        if (len(new_X) > 0) and (len(new_U) > 0) and (len(new_Y) > 0):
            if self.restart_on_loop:
                # Get rid of continuity between trajs since restart
                self.X = np.concatenate((self.X, reshape_pt1(new_X)), axis=0)
                self.U = np.concatenate((self.U, reshape_pt1(new_U)), axis=0)
                self.Y = np.concatenate((self.Y, reshape_pt1(new_Y)), axis=0)
                whole_X = np.concatenate((whole_X, reshape_pt1(new_X)), axis=0)
                whole_U = np.concatenate((whole_U, reshape_pt1(new_U)), axis=0)
                whole_Y = np.concatenate((whole_Y, reshape_pt1(new_Y)), axis=0)
            else:
                # Get rid of last point of previous traj since no restart
                self.X = np.concatenate((self.X[:-1, :], reshape_pt1(new_X)),
                                        axis=0)
                self.U = np.concatenate((self.U[:-1, :], reshape_pt1(new_U)),
                                        axis=0)
                self.Y = np.concatenate((self.Y[:-1, :], reshape_pt1(new_Y)),
                                        axis=0)
                whole_X = np.concatenate((whole_X[:-1, :], reshape_pt1(new_X)),
                                         axis=0)
                whole_U = np.concatenate((whole_U[:-1, :], reshape_pt1(new_U)),
                                         axis=0)
                whole_Y = np.concatenate((whole_Y[:-1, :], reshape_pt1(new_Y)),
                                         axis=0)

            # Save new data to separate csv
            filename = 'new_X' + '.csv'
            file = pd.DataFrame(new_X)
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            filename = 'new_U' + '.csv'
            file = pd.DataFrame(new_U)
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
            filename = 'new_Y' + '.csv'
            file = pd.DataFrame(new_Y)
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
        elif ((len(new_X) > 0) or (len(new_U) > 0) or (
                len(new_Y) > 0)) and not (
                (len(new_X) > 0) or (len(new_U) > 0) or (len(new_Y) > 0)):
            raise ValueError(
                'Only partial new data has been given to re-train the GP. '
                'Please make sure you enter new X, U and Y.')
        self.sample_idx = len(whole_X)  # Nb of samples since start
        pure_X = whole_X.copy()
        pure_U = whole_U.copy()
        pure_Y = whole_Y.copy()

        if self.sliding_window_size:
            whole_X = whole_X[-self.sliding_window_size:, :]
            whole_U = whole_U[-self.sliding_window_size:, :]
            whole_Y = whole_Y[-self.sliding_window_size:, :]

        if self.prior_mean:
            # Only consider residuals = Y - prior_mean(X,U) as output data
            prior_mean_vector = self.prior_mean(whole_X, whole_U,
                                                self.prior_kwargs)
            if self.monitor_experiment:
                for i in range(prior_mean_vector.shape[1]):
                    name = 'GP_prior_mean' + str(i) + '.pdf'
                    plt.plot(prior_mean_vector[:, i],
                             label='Prior mean ' + str(i))
                    plt.plot(whole_Y[:, i], label='Output data ' + str(i))
                    plt.title('Visualization of prior mean given to GP')
                    plt.legend()
                    plt.xlabel('Time step')
                    plt.ylabel('Prior')
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    if self.verbose:
                        plt.show()
                    plt.close('all')
            whole_Y = reshape_pt1(whole_Y - prior_mean_vector)

        self.scaler_X = preprocessing.StandardScaler().fit(whole_X)
        whole_scaled_X = self.scaler_X.transform(whole_X)
        if self.no_control:
            # Ignore control for learning
            self.GP_X = whole_scaled_X
        else:
            self.GP_X = np.concatenate((whole_scaled_X, whole_U), axis=1)
        self.scaler_Y = preprocessing.StandardScaler().fit(whole_Y)
        self.GP_Y = self.scaler_Y.transform(whole_Y)
        return pure_X, pure_U, pure_Y

    def update_model(self):
        # Create model and optimize hyperparameters
        self.create_GP_model()
        self.optimize_hyperparameters(self.model)
        if self.sparse and not (self.nb_inducing_inputs >= len(self.GP_X)):
            start = len(self.model.param_array) - (len(
                self.model.kern.param_array) + 1)
            end = len(self.model.param_array)
            if not self.hyperparams.any():
                self.hyperparams = reshape_pt1(np.array(
                    [self.sample_idx] + [np.copy(self.model.param_array)[i]
                                         for i in range(start, end)]))
            else:
                self.hyperparams = np.concatenate(
                    (self.hyperparams, reshape_pt1(
                        np.array([self.sample_idx] + [
                            np.copy(self.model.param_array)[i] for i in
                            range(start, end)]))), axis=0)
            if self.monitor_experiment:
                self.GP_inducing_inputs = reshape_dim1([np.copy(
                    self.model.inducing_inputs)[i] for i in range(len(
                    self.model.inducing_inputs))])
                df = pd.DataFrame(self.GP_inducing_inputs)
                df.to_csv(os.path.join(self.results_folder,
                                       'Z_GP_inducing_inputs.csv'),
                          header=False)

        else:
            if not self.hyperparams.any():
                self.hyperparams = reshape_pt1(np.array(
                    [self.sample_idx] + [np.copy(self.model.param_array)[i]
                                         for i in range(
                            len(self.model.param_array))]))
            else:
                self.hyperparams = np.concatenate(
                    (self.hyperparams, reshape_pt1(
                        np.array([self.sample_idx] + [
                            np.copy(self.model.param_array)[i] for i in
                            range(len(self.model.param_array))]))), axis=0)
        if self.monitor_experiment:
            for i in range(self.GP_X.shape[1]):
                for j in range(self.GP_Y.shape[1]):
                    # Plot GP model at each input dim i over np.arange(dataMax,
                    # dataMin) while all other input dims are set at 0, j =
                    # output dim as vertical axis. Predicitons are valid
                    # inisde GP model (no prior mean taken into account,
                    # normalized data)!!
                    name = 'GP_model_update' + str(i) + str(j) + '.pdf'
                    self.model.plot(plot_density=False, plot_data=False,
                                    visible_dims=[i], which_data_ycols=[j])
                    plt.title('Visualization of GP posterior')
                    plt.legend()
                    plt.xlabel('Input state ' + str(i))
                    plt.ylabel('GP prediction ' + str(j))
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    if self.verbose:
                        plt.show()
                    plt.close('all')

    def optimize_hyperparameters(self, model):
        # Optimize hyperparameters
        if self.hyperparam_optim == 'fixed_hyperparameters':
            pass
        elif self.hyperparam_optim == 'fixed_start':
            model.optimize(start=self.start_optim_fixed, messages=self.verbose,
                           max_iters=100)
        else:

            model.optimize_restarts(num_restarts=5, verbose=True,
                                    max_iters=100, robust=True)
        self.last_optimparams = self.step
        if self.verbose:
            logging.info(model)

    def evaluate_model(self):
        # Record RMSE, average log_likelihood, standardized versions
        if self.memory_saving:
            # Read grid, true prediction and controls from csv
            grid = pd.read_csv(
                os.path.join(self.results_folder, 'Evaluation_grid.csv'),
                sep=',', header=None)
            grid = grid.drop(grid.columns[0], axis=1).values
            grid_controls = pd.read_csv(
                os.path.join(self.results_folder, 'Grid_controls.csv'),
                sep=',', header=None)
            grid_controls = grid_controls.drop(grid_controls.columns[0],
                                               axis=1).values
            true_predicted_grid = pd.read_csv(
                os.path.join(self.results_folder, 'True_predicted_grid.csv'),
                sep=',', header=None)
            true_predicted_grid = true_predicted_grid.drop(
                true_predicted_grid.columns[0], axis=1).values
            self.grid = np.reshape(grid, (-1, self.X.shape[1]))
            self.grid_controls = np.reshape(grid_controls,
                                            (-1, self.U.shape[1]))
            self.true_predicted_grid = np.reshape(true_predicted_grid,
                                                  (-1, self.Y.shape[1]))

        RMSE_array_dim, RMSE, SRMSE, predicted_grid, true_predicted_grid, \
        grid_controls, log_likelihood, stand_log_likelihood = \
            self.compute_l2error_grid(
                grid=self.grid, grid_controls=self.grid_controls,
                true_predicted_grid=self.true_predicted_grid)
        self.RMSE_time = np.concatenate((self.RMSE_time, reshape_pt1(
            np.array([self.sample_idx, RMSE]))), axis=0)
        self.SRMSE_time = np.concatenate((self.SRMSE_time, reshape_pt1(
            np.array([self.sample_idx, SRMSE]))), axis=0)
        self.log_likelihood_time = np.concatenate(
            (self.log_likelihood_time, reshape_pt1(
                np.array([self.sample_idx, log_likelihood]))), axis=0)
        self.stand_log_likelihood_time = np.concatenate(
            (self.stand_log_likelihood_time, reshape_pt1(
                np.array([self.sample_idx, stand_log_likelihood]))), axis=0)

        # Write evaluation results to a file
        filename = 'Predicted_grid' + str(self.step) + '.csv'
        file = pd.DataFrame(predicted_grid)
        file.to_csv(os.path.join(self.results_folder, filename), header=False)
        if self.step > 1 and self.memory_saving:
            # Delete grid and controls again
            self.grid = self.grid[-1:, :]
            self.grid_controls = self.grid_controls[-1:, :]
            self.true_predicted_grid = self.true_predicted_grid[-1:, :]
        return RMSE_array_dim, RMSE, SRMSE, predicted_grid, \
               true_predicted_grid, grid_controls, log_likelihood, \
               stand_log_likelihood

    def predict(self, x, u, scale=True, only_prior=False):
        # Predict outcome of input, adding prior mean if necessary
        if self.multioutput_GP:
            logging.error('Not implemented yet for multioutput GPs')
        else:
            x = reshape_pt1(x)
            u = reshape_pt1(u)
            if scale:
                scaled_x = reshape_pt1(self.scaler_X.transform(x))
                unscaled_x = reshape_pt1(x)
            else:
                scaled_x = reshape_pt1(x)
                unscaled_x = reshape_pt1(self.scaler_X.inverse_transform(x))
            if only_prior:
                if not self.prior_mean:
                    raise ValueError('No prior mean given with only_prior '
                                     'option.')
                # Return prior mean: predicted_x = prior(x)
                prior = reshape_pt1(
                    self.prior_mean(unscaled_x, u, self.prior_kwargs))
                predicted_mean = prior
                predicted_lowconf = prior
                predicted_uppconf = prior
                predicted_var = reshape_pt1([0])
            else:
                if self.no_control:
                    # Ignore control for learning
                    GP_x = reshape_pt1(scaled_x)
                else:
                    GP_x = np.concatenate((reshape_pt1(scaled_x),
                                           reshape_pt1(u)), axis=1)
                mean, var = self.model.predict(GP_x, full_cov=False)
                predicted_mean = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean))
                predicted_var = reshape_pt1(var)
                predicted_lowconf = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean - 2 * var))
                predicted_uppconf = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean + 2 * var))
                if self.prior_mean:
                    # Add prior mean: predicted_x = GP_predicts(x) + prior(x)
                    prior = reshape_pt1(
                        self.prior_mean(unscaled_x, u, self.prior_kwargs))
                    predicted_mean = predicted_mean + prior
                    predicted_lowconf = predicted_lowconf + prior
                    predicted_uppconf = predicted_uppconf + prior
            return predicted_mean, predicted_var, predicted_lowconf, \
                   predicted_uppconf

    def predict_deriv(self, x, u, scale=True, only_x=False, only_prior=False):
        # Predict derivative of posterior distribution d mean / dx, d var /
        # dx at x, adding prior mean if necessary
        if self.multioutput_GP:
            logging.error('Not implemented yet for multioutput GPs')
        else:
            x = reshape_pt1(x)
            u = reshape_pt1(u)
            if scale:
                scaled_x = reshape_pt1(self.scaler_X.transform(x))
                unscaled_x = reshape_pt1(x)
            else:
                scaled_x = reshape_pt1(x)
                unscaled_x = reshape_pt1(self.scaler_X.inverse_transform(x))
            if only_prior:
                # Return prior mean: predicted_x = prior(x)
                prior = reshape_pt1(
                    self.prior_mean(unscaled_x, u, self.prior_kwargs))
                predicted_mean = prior
                predicted_lowconf = prior
                predicted_uppconf = prior
                predicted_var = reshape_pt1([0])
            else:
                if self.no_control:
                    # Ignore control for learning
                    GP_x = reshape_pt1(scaled_x)
                else:
                    GP_x = np.concatenate(
                        (reshape_pt1(scaled_x), reshape_pt1(u)),
                        axis=1)
                mean, var = self.model.predictive_gradients(GP_x, self.kernel)
                mean = np.reshape(mean, GP_x.shape)
                var = np.reshape(var, GP_x.shape)
                predicted_mean = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean))
                predicted_var = reshape_pt1(var)
                predicted_lowconf = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean - 2 * var))
                predicted_uppconf = reshape_pt1(
                    self.scaler_Y.inverse_transform(mean + 2 * var))
                if self.prior_mean:
                    assert self.prior_mean_deriv, 'A prior function was given ' \
                                                  'and used in the mean, but its ' \
                                                  'derivative was not given for ' \
                                                  'the derivative predictions.'
                if self.prior_mean_deriv:
                    # Add prior mean: predicted_x = GP_predicts(x) + prior(x)
                    prior = reshape_pt1(
                        self.prior_mean_deriv(unscaled_x, u, self.prior_kwargs))
                    predicted_mean = predicted_mean + prior
                    predicted_lowconf = predicted_lowconf + prior
                    predicted_uppconf = predicted_uppconf + prior
            if only_x:
                return predicted_mean[:, :x.shape[1]], \
                       predicted_var[:, :x.shape[1]], \
                       predicted_lowconf[:, :x.shape[1]], \
                       predicted_uppconf[:, :x.shape[1]]
            else:
                return predicted_mean, predicted_var, predicted_lowconf, \
                       predicted_uppconf

    def predict_euler_Michelangelo(self, x, u, scale=True, only_prior=False):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        # Predict x_t+1 from prediction of y_t in Michelangelo framework
        # TODO better than Euler?
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = np.eye(x.shape[1], k=1)
        B = np.zeros((x.shape[1], 1))
        B[-1] = 1
        ABmult = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(predicted_mean))
        ABmult_lowconf = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(predicted_lowconf))
        ABmult_uppconf = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(predicted_uppconf))
        predicted_mean_euler = reshape_pt1(x) + self.dt * (ABmult)  # + u)
        predicted_mean_euler_lowconf = reshape_pt1(x) + self.dt * (
            ABmult_lowconf)  # + u)
        predicted_mean_euler_uppconf = reshape_pt1(x) + self.dt * (
            ABmult_uppconf)  # + u)
        return predicted_mean_euler, predicted_var, \
               predicted_mean_euler_lowconf, predicted_mean_euler_uppconf

    def true_dynamics_euler_Michelangelo(self, x, u):
        # Compute x_t+1 from true y_t in Michelangelo framework
        # TODO better than Euler?
        true_mean = self.true_dynamics(x, u)
        A = np.eye(x.shape[1], k=1)
        B = np.zeros((x.shape[1], 1))
        B[-1] = 1
        ABmult = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(true_mean))
        true_mean_euler = reshape_pt1(x) + self.dt * (ABmult)  # + u)
        return true_mean_euler

    def predict_euler_discrete_justvelocity(self, x, u, scale=True,
                                            only_prior=False):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        # Predict x_t+1 from prediction of velocity_t with chain of integrators
        # TODO better than Euler?
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = np.eye(x.shape[1]) + self.dt * np.eye(x.shape[1], k=1)
        A[-1] = np.zeros((1, x.shape[1]))
        B = np.zeros((x.shape[1], 1))
        B[-1] = 1
        predicted_mean_euler = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(predicted_mean))
        predicted_mean_euler_lowconf = np.dot(A, reshape_pt1_tonormal(x)) + \
                                       np.dot(B, reshape_pt1_tonormal(
                                           predicted_lowconf))
        predicted_mean_euler_uppconf = np.dot(A, reshape_pt1_tonormal(x)) + \
                                       np.dot(B, reshape_pt1_tonormal(
                                           predicted_uppconf))
        return reshape_pt1(predicted_mean_euler), reshape_pt1(predicted_var), \
               reshape_pt1(predicted_mean_euler_lowconf), \
               reshape_pt1(predicted_mean_euler_uppconf)

    def true_dynamics_euler_discrete_justvelocity(self, x, u):
        # Compute x_t+1 from true velocity_t with chain of integrators
        # TODO better than Euler?
        true_mean = self.true_dynamics(x, u)
        A = np.eye(x.shape[1]) + self.dt * np.eye(x.shape[1], k=1)
        A[-1] = np.zeros((1, x.shape[1]))
        B = np.zeros((x.shape[1], 1))
        B[-1] = 1
        true_mean_euler = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(true_mean))
        return reshape_pt1(true_mean_euler)

    def predict_euler_continuous_justvelocity(self, x, u, scale=True,
                                              only_prior=False):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        # Predict x_t+1 from prediction of xdot_t with chain of integrators
        # TODO better than Euler?
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = self.predict(x, u, scale=scale,
                                         only_prior=only_prior)
        A = np.eye(x.shape[1]) + self.dt * np.eye(x.shape[1], k=1)
        B = np.zeros((x.shape[1], 1))
        B[-1] = self.dt
        predicted_mean_euler = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(predicted_mean))
        predicted_mean_euler_lowconf = np.dot(A, reshape_pt1_tonormal(x)) + \
                                       np.dot(B, reshape_pt1_tonormal(
                                           predicted_lowconf))
        predicted_mean_euler_uppconf = np.dot(A, reshape_pt1_tonormal(x)) + \
                                       np.dot(B, reshape_pt1_tonormal(
                                           predicted_uppconf))
        return reshape_pt1(predicted_mean_euler), reshape_pt1(predicted_var), \
               reshape_pt1(predicted_mean_euler_lowconf), \
               reshape_pt1(predicted_mean_euler_uppconf)

    def true_dynamics_euler_continuous_justvelocity(self, x, u):
        # Compute x_t+1 from true xdot_t with chain of integrators
        # TODO better than Euler?
        true_mean = self.true_dynamics(x, u)
        A = np.eye(x.shape[1]) + self.dt * np.eye(x.shape[1], k=1)
        B = np.zeros((x.shape[1], 1))
        B[-1] = self.dt
        true_mean_euler = np.dot(A, reshape_pt1_tonormal(x)) + np.dot(
            B, reshape_pt1_tonormal(true_mean))
        return reshape_pt1(true_mean_euler)

    def compute_l2error_grid(self, grid, grid_controls, true_predicted_grid,
                             use_euler=None):
        predicted_grid = np.concatenate((np.copy(true_predicted_grid),
                                         np.zeros((true_predicted_grid.
                                                   shape[0], 1))), axis=1)
        # RMSE, at fixed control, between real and GP predictions
        # Average log probability of real prediction coming from GP predicted
        # distribution (in scaled domain)
        l2_error_array = np.zeros(
            (len(true_predicted_grid), true_predicted_grid.shape[1]))
        for idx, x in enumerate(grid):
            control = reshape_pt1(grid_controls[idx])
            if not use_euler:
                predicted_mean, predicted_var, predicted_lowconf, \
                predicted_uppconf = self.predict(x, control)
            elif use_euler == 'Michelangelo':
                predicted_mean, predicted_var, predicted_lowconf, \
                predicted_uppconf = \
                    self.predict_euler_Michelangelo(x, control)
            elif use_euler == 'discrete_justvelocity':
                predicted_mean, predicted_var, predicted_lowconf, \
                predicted_uppconf = \
                    self.predict_euler_discrete_justvelocity(x, control)
            elif use_euler == 'continuous_justvelocity':
                predicted_mean, predicted_var, predicted_lowconf, \
                predicted_uppconf = \
                    self.predict_euler_continuous_justvelocity(x, control)
            else:
                logging.error('This version of Euler/discretized prediction '
                              'is not implemented.')
            true_mean = reshape_pt1(true_predicted_grid[idx])
            predicted_grid[idx, :-1] = predicted_mean
            predicted_grid[idx, -1] = predicted_var

            l2_error_array[idx] = np.square(
                reshape_pt1(true_mean) - predicted_mean)
        self.variables['l2_error_array'] = l2_error_array

        RMSE_array_dim = np.mean(l2_error_array, axis=0)
        RMSE = RMS(true_predicted_grid - predicted_grid[:, :-1])
        if true_predicted_grid.shape[1] > 1:
            var = np.linalg.det(np.cov(true_predicted_grid.T))
        else:
            var = np.var(true_predicted_grid)
        SRMSE = RMSE / var
        log_likelihood = log_multivariate_normal_likelihood(
            true_predicted_grid, predicted_grid[:, :-1],
            predicted_grid[:, -1])
        if reshape_pt1(self.scaler_Y.mean_).shape[1] == \
                true_predicted_grid.shape[1]:
            mean_vector = reshape_pt1(
                np.repeat(reshape_pt1(self.scaler_Y.mean_),
                          len(true_predicted_grid), axis=0))
            var_vector = reshape_pt1(np.repeat(reshape_pt1(self.scaler_Y.var_),
                                               len(true_predicted_grid),
                                               axis=0))
        else:
            mean_vector = reshape_pt1(
                np.repeat(reshape_pt1(self.scaler_X.mean_),
                          len(true_predicted_grid), axis=0))
            var_vector = reshape_pt1(np.repeat(reshape_pt1(self.scaler_X.var_),
                                               len(true_predicted_grid),
                                               axis=0))
        stand_log_likelihood = log_likelihood - \
                               log_multivariate_normal_likelihood(
                                   true_predicted_grid, mean_vector,
                                   var_vector)

        return RMSE_array_dim, RMSE, SRMSE, predicted_grid, \
               true_predicted_grid, grid_controls, log_likelihood, \
               stand_log_likelihood

    def create_grid(self, init_state, init_control, constrain_u, grid_inf,
                    grid_sup):
        # Create random grid for evaluation
        # https://stackoverflow.com/questions/45583274/how-to-generate-an-n-dimensional-grid-in-python
        dx = init_state.shape[1]
        du = init_control.shape[1]
        if constrain_u:
            umin = np.min(constrain_u)
            umax = np.max(constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = np.min(self.U, axis=0)
            umax = np.max(self.U, axis=0)
        nb_points = int(np.ceil(np.max([10 ** 4 / dx, 500])))
        if not self.ground_truth_approx:
            grid = reshape_pt1(np.random.uniform(grid_inf, grid_sup,
                                                 size=(nb_points, dx)))
            grid_controls = reshape_pt1(np.random.uniform(umin, umax,
                                                          size=(nb_points, du)))
        else:
            grid = reshape_pt1(self.X)
            grid_controls = reshape_pt1(self.U)
        return grid, grid_controls

    def create_true_predicted_grid(self, grid, grid_controls):
        true_predicted_grid = np.zeros((len(grid), self.Y.shape[1]))
        if not self.ground_truth_approx:
            for idx, x in enumerate(true_predicted_grid):
                control = reshape_pt1(grid_controls[idx])
                x = reshape_pt1(grid[idx])
                true_predicted_grid[idx] = self.true_dynamics(x, control)
        else:
            true_predicted_grid = reshape_pt1(self.Y)
        return true_predicted_grid

    def read_grid_variables(self, results_folder):
        for key, val in self.grid_variables.items():
            filename = str(key) + '.csv'
            data = pd.read_csv(
                os.path.join(results_folder, filename), sep=',', header=None)
            self.grid_variables[str(key)] = data.drop(data.columns[0],
                                                      axis=1).values

    def save_grid_variables(self, grid, grid_controls, true_predicted_grid,
                            results_folder):
        grid = pd.DataFrame(grid)
        grid.to_csv(os.path.join(results_folder, 'Evaluation_grid.csv'),
                    header=False)
        grid_controls = pd.DataFrame(grid_controls)
        grid_controls.to_csv(os.path.join(results_folder, 'Grid_controls.csv'),
                             header=False)
        true_predicted_grid = pd.DataFrame(true_predicted_grid)
        true_predicted_grid.to_csv(
            os.path.join(results_folder, 'True_predicted_grid.csv'),
            header=False)
        if self.memory_saving:
            self.grid = self.grid[-1:, :]
            self.grid_controls = self.grid_controls[-1:, :]
            self.true_predicted_grid = self.true_predicted_grid[-1:, :]

    def create_rollout_list(self):
        rollout_list = []
        if self.constrain_u:
            umin = np.min(self.constrain_u)
            umax = np.max(self.constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = np.min(self.U, axis=0)
            umax = np.max(self.U, axis=0)
        # Quite slow, parallelize a bit?
        for controller, current_nb_rollouts in self.rollout_controller.items():
            i = 0
            while i < current_nb_rollouts:
                if not self.ground_truth_approx:
                    time_vector = np.arange(0, self.rollout_length) * self.dt
                    init_state = reshape_pt1(np.random.uniform(
                        self.grid_inf, self.grid_sup,
                        size=(1, self.X.shape[1])))
                    true_mean = np.zeros((self.rollout_length + 1,
                                          init_state.shape[1]))
                    true_mean[0] = init_state.copy()
                    # Define control_traj depending on current controller
                    if controller == 'random':
                        control_traj = reshape_dim1(
                            np.random.uniform(umin, umax, size=(
                                self.rollout_length, self.U.shape[1])))
                    elif controller == 'sin_controller_1D':
                        control_traj = sin_controller_1D(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    elif controller == 'sin_controller_02D':
                        control_traj = sin_controller_02D(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    elif controller == 'null_controller':
                        control_traj = null_controller(
                            time_vector, self.config, self.t0, reshape_pt1(
                                self.U[0]))
                    else:
                        raise ValueError(
                            'Controller for rollout is not defined. Available '
                            'options are random, sin_controller_1D, '
                            'sin_controller_02D, null_controller.')
                    for t in range(self.rollout_length):
                        # True and predicted trajectory over time
                        control = control_traj[t]
                        if 'Michelangelo' in self.system:
                            xnext_true = self.true_dynamics_euler_Michelangelo(
                                reshape_pt1(
                                    true_mean[t]), reshape_pt1(control))
                        elif ('justvelocity' in self.system) and not \
                                self.continuous_model:
                            xnext_true = \
                                self.true_dynamics_euler_discrete_justvelocity(
                                    reshape_pt1(true_mean[t]),
                                    reshape_pt1(control))
                        elif 'justvelocity' in self.system and \
                                self.continuous_model:
                            xnext_true = \
                                self.true_dynamics_euler_continuous_justvelocity(
                                    reshape_pt1(true_mean[t]),
                                    reshape_pt1(control))
                        else:
                            xnext_true = self.true_dynamics(reshape_pt1(
                                true_mean[t]), reshape_pt1(control))
                        true_mean[t + 1] = xnext_true
                    max = np.max(np.abs(true_mean))
                    if max > self.max_rollout_value:
                        # If true trajectory diverges, ignore this rollout
                        logging.warning(
                            'Ignored a rollout with diverging true '
                            'trajectory, with initial state ' + str(
                                init_state) + ' and maximum reached abs value '
                            + str(max))
                        continue
                    i += 1
                    rollout_list.append([init_state, control_traj, true_mean])
                else:
                    if self.step > 0:
                        # Only initialize rollout list at beginning of each fold
                        return self.rollout_list
                    # If no ground truth, rollouts are subsets of train data
                    if i == 0:
                        # Initial rollout same as data
                        init_state = reshape_pt1(self.init_state)
                        true_mean = reshape_pt1(
                            self.X[:self.rollout_length + 1, :])
                        control_traj = reshape_pt1(
                            self.U[:self.rollout_length, :])
                    else:
                        # Next rollouts start, control random subset of data
                        start_idx = np.random.randint(0, len(self.U) -
                                                      self.rollout_length)
                        init_state = reshape_pt1(self.X[start_idx])
                        true_mean = reshape_pt1(
                            self.X[start_idx:start_idx +
                                             self.rollout_length + 1, :])
                        control_traj = reshape_pt1(
                            self.U[start_idx:start_idx +
                                             self.rollout_length, :])
                    rollout_list.append([init_state, control_traj, true_mean])
                    i += 1
        return np.array(rollout_list, dtype=object)

    def read_rollout_list(self, results_folder, nb_rollouts, step,
                          folder_title=None):
        rollout_list = []
        for i in range(nb_rollouts):
            if not folder_title:
                folder = os.path.join(results_folder, 'Rollouts_' + str(step))
            else:
                folder = os.path.join(results_folder, folder_title + '_' +
                                      str(step))
            rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
            filename = 'Init_state.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            init_state = data.drop(data.columns[0], axis=1).values
            filename = 'Control_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            control_traj = data.drop(data.columns[0], axis=1).values
            filename = 'True_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            true_mean = data.drop(data.columns[0], axis=1).values
            rollout_list.append([init_state, control_traj, true_mean])
        return np.array(rollout_list, dtype=object)

    def evaluate_rollouts(self, only_prior=False):
        if len(self.rollout_list) == 0:
            return 0
        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         only_prior=only_prior)
        self.specs['nb_rollouts'] = self.nb_rollouts
        self.specs['rollout_length'] = self.rollout_length
        self.specs['rollout_RMSE'] = rollout_RMSE
        self.specs['rollout_SRMSE'] = rollout_SRMSE
        self.specs['rollout_log_AL'] = rollout_log_AL
        self.specs['rollout_stand_log_AL'] = rollout_stand_log_AL
        self.rollout_RMSE = \
            np.concatenate((self.rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.rollout_SRMSE = \
            np.concatenate((self.rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.rollout_log_AL = \
            np.concatenate((self.rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.rollout_stand_log_AL = \
            np.concatenate((self.rollout_stand_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['rollout_RMSE'] = self.rollout_RMSE
        self.variables['rollout_SRMSE'] = self.rollout_SRMSE
        self.variables['rollout_log_AL'] = self.rollout_log_AL
        self.variables['rollout_stand_log_AL'] = self.rollout_stand_log_AL
        plot_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = np.concatenate(
                (self.rollout_list, rollout_list), axis=1)
            save_rollout_variables(
                self.results_folder, self.nb_rollouts, complete_rollout_list,
                step=self.step - 1, results=True,
                ground_truth_approx=self.ground_truth_approx)

    def evaluate_kalman_rollouts(self, observer, observe_data,
                                 discrete_observer, no_GP_in_observer=False,
                                 only_prior=False):
        if len(self.rollout_list) == 0:
            return 0

        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory, but in closed loop
        # by correcting with observations at each time step
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         observer=observer, observe_data=observe_data,
                         discrete_observer=discrete_observer, kalman=True,
                         no_GP_in_observer=no_GP_in_observer,
                         only_prior=only_prior)
        self.specs['kalman_rollout_RMSE'] = rollout_RMSE
        self.specs['kalman_rollout_SRMSE'] = rollout_SRMSE
        self.specs['kalman_rollout_log_AL'] = rollout_log_AL
        self.specs['kalman_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.kalman_rollout_RMSE = \
            np.concatenate((self.kalman_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.kalman_rollout_SRMSE = \
            np.concatenate((self.kalman_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.kalman_rollout_log_AL = \
            np.concatenate((self.kalman_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.kalman_rollout_stand_log_AL = \
            np.concatenate((self.kalman_rollout_stand_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['kalman_rollout_RMSE'] = self.kalman_rollout_RMSE
        self.variables['kalman_rollout_SRMSE'] = \
            self.kalman_rollout_SRMSE
        self.variables['kalman_rollout_log_AL'] = \
            self.kalman_rollout_log_AL
        self.variables['kalman_rollout_stand_log_AL'] = \
            self.kalman_rollout_stand_log_AL
        plot_kalman_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = np.concatenate(
                (self.rollout_list, rollout_list), axis=1)
            save_kalman_rollout_variables(
                self.results_folder, self.nb_rollouts, complete_rollout_list,
                step=self.step - 1,
                ground_truth_approx=self.ground_truth_approx)

    def evaluate_closedloop_rollouts(self, observer, observe_data,
                                     no_GP_in_observer=False):
        if len(self.rollout_list) == 0:
            return 0

        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory, but in closed loop
        # by correcting with observations at each time step
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
        rollout_stand_log_AL = \
            run_rollouts(self, self.rollout_list, folder=self.results_folder,
                         observer=observer, observe_data=observe_data,
                         closedloop=True, no_GP_in_observer=no_GP_in_observer)
        self.specs['closedloop_rollout_RMSE'] = rollout_RMSE
        self.specs['closedloop_rollout_SRMSE'] = rollout_SRMSE
        self.specs['closedloop_rollout_log_AL'] = rollout_log_AL
        self.specs['closedloop_rollout_stand_log_AL'] = rollout_stand_log_AL
        self.closedloop_rollout_RMSE = \
            np.concatenate((self.closedloop_rollout_RMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_RMSE]))), axis=0)
        self.closedloop_rollout_SRMSE = \
            np.concatenate((self.closedloop_rollout_SRMSE, reshape_pt1(
                np.array([self.sample_idx, rollout_SRMSE]))), axis=0)
        self.closedloop_rollout_log_AL = \
            np.concatenate((self.closedloop_rollout_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_log_AL]))), axis=0)
        self.closedloop_rollout_stand_log_AL = np.concatenate((
            self.closedloop_rollout_stand_log_AL, reshape_pt1(
                np.array([self.sample_idx, rollout_stand_log_AL]))), axis=0)
        self.variables['closedloop_rollout_RMSE'] = \
            self.closedloop_rollout_RMSE
        self.variables['closedloop_rollout_SRMSE'] = \
            self.closedloop_rollout_SRMSE
        self.variables['closedloop_rollout_log_AL'] = \
            self.closedloop_rollout_log_AL
        self.variables['closedloop_rollout_stand_log_AL'] = \
            self.closedloop_rollout_stand_log_AL
        plot_closedloop_rollout_data(self, folder=self.results_folder)
        if self.nb_rollouts > 0:
            complete_rollout_list = np.concatenate(
                (self.rollout_list, rollout_list), axis=1)
            save_closedloop_rollout_variables(
                self.results_folder, self.nb_rollouts, complete_rollout_list,
                step=self.step - 1,
                ground_truth_approx=self.ground_truth_approx)

    def save_log(self, results_folder):
        logging.INFO
        logging.FileHandler(
            "{0}/{1}.log".format(results_folder,
                                 'log' + str(sys.argv[1])))
        logging.basicConfig(level=logging.INFO)

    def read_variables(self, results_folder):
        for key, val in self.variables.items():
            if key.startswith('test_') or key.startswith('val_') or (
                    'rollout' in key):
                # Avoid saving all test and validation variables, and only
                # save rollout variables in rollout functions
                continue
            filename = str(key) + '.csv'
            data = pd.read_csv(
                os.path.join(results_folder, filename), sep=',', header=None)
            self.variables[str(key)] = data.drop(data.columns[0], axis=1).values

    def read_anim_variables(self, results_folder):
        for key, val in self.variables.items():
            if (key == 'X') or (key == 'Y'):
                filename = str(key) + '.csv'
                data = pd.read_csv(
                    os.path.join(results_folder, filename), sep=',',
                    header=None)
                self.variables[str(key)] = data.drop(data.columns[0],
                                                     axis=1).values

    def read_control(self, results_folder):
        for key, val in self.variables.items():
            if (key == 'U'):
                filename = str(key) + '.csv'
                data = pd.read_csv(
                    os.path.join(results_folder, filename), sep=',',
                    header=None)
                self.variables[str(key)] = data.drop(data.columns[0],
                                                     axis=1).values

    def cut_anim_variables(self):
        for key, val in self.variables.items():
            if (key == 'X') or (key == 'Y'):
                self.variables[key] = self.variables[key][-1:, :]

    def set_results_folder(self, folder):
        # Change results folder and copy grid variables and log there
        self.read_grid_variables(self.results_folder)
        self.save_grid_variables(self.grid, self.grid_controls,
                                 self.true_predicted_grid, folder)
        self.rollout_list = self.read_rollout_list(self.results_folder,
                                                   self.nb_rollouts,
                                                   step=self.step - 1)
        save_rollout_variables(folder, self.nb_rollouts, self.rollout_list,
                               step=self.step,
                               ground_truth_approx=self.ground_truth_approx)
        self.save_log(folder)
        self.results_folder = folder

    def set_dyn_kwargs(self, new_dyn_kwargs):
        # Change dyn kwargs
        self.dyn_kwargs = new_dyn_kwargs
        self.specs['dyn_kwargs'] = new_dyn_kwargs
        self.specs.update(new_dyn_kwargs)

    def set_config(self, new_config: Config):
        # Change config
        self.config = new_config
        self.specs.update(new_config.params)

    def save_intermediate(self, memory_saving=False, ignore_first=False):
        # Retrieve and save all intermediate variables
        self.variables['X'] = self.X
        self.variables['Y'] = self.Y
        self.variables['U'] = self.U
        self.variables['Computation_time'] = np.array(self.time)
        self.specs['sparse'] = self.sparse
        if self.monitor_experiment:
            self.variables['RMSE_time'] = np.array(self.RMSE_time)
            self.variables['SRMSE_time'] = np.array(self.SRMSE_time)
            self.variables['log_AL_time'] = np.array(self.log_likelihood_time)
            self.variables['stand_log_AL_time'] = np.array(
                self.stand_log_likelihood_time)

        # Store results and parameters in files
        if self.step > 1:
            specs_file = os.path.join(self.results_folder, 'Specifications.txt')
            with open(specs_file, 'w') as f:
                for key, val in self.specs.items():
                    if key == 'kernel':
                        if self.multioutput_GP:
                            for k in range(self.model.nb_output_dims):
                                for i in range(len(
                                        self.model.models[k].kern.parameters)):
                                    print(
                                        self.model.models[k].kern.parameters[i],
                                        file=f)
                        else:
                            for i in range(len(self.model.kern.parameters)):
                                print(self.model.kern.parameters[i], file=f)
                    else:
                        print(key, ': ', val, file=f)
        for key, val in self.variables.items():
            if key.startswith('test_') or key.startswith('val_') or (
                    'rollout' in key):
                # Avoid saving all test and validation variables, and only
                # save rollout variables in rollout functions
                continue
            filename = str(key) + '.csv'
            if memory_saving:
                # Append only new values to csv
                if self.step > 1:
                    if ignore_first:
                        file = pd.DataFrame(val[1:, :])
                    else:
                        file = pd.DataFrame(val)
                else:
                    file = pd.DataFrame(val)
                file.to_csv(os.path.join(self.results_folder, filename),
                            mode='a', header=False)
            else:
                file = pd.DataFrame(val)
                file.to_csv(os.path.join(self.results_folder, filename),
                            header=False)

        if memory_saving:
            # Keep only last value in variable
            for key, val in self.variables.items():
                self.variables[key] = self.variables[key][-1:, :]
            for key, val in self.grid_variables.items():
                self.grid_variables[key] = self.grid_variables[key][-1:, :]
            self.X = self.X[-1:, :]
            self.Y = self.Y[-1:, :]
            self.U = self.U[-1:, :]
            self.time = self.time[-1:, :]
            if self.monitor_experiment:
                self.RMSE_time = self.RMSE_time[-1:, :]
                self.SRMSE_time = self.SRMSE_time[-1:, :]
                self.log_likelihood_time = self.log_likelihood_time[-1:, :]
                self.stand_log_likelihood_time = self.stand_log_likelihood_time[
                                                 -1:, :]

        if self.verbose:
            logging.info('Saved intermediate results in')
            logging.info(self.results_folder)

    def save_folder(self, results_folder):
        # Save all variables in a folder, plot results over time
        self.variables['X'] = self.X
        self.variables['Y'] = self.Y
        self.variables['U'] = self.U
        self.specs['model'] = self.model
        # Store parameters in file
        specs_file = os.path.join(self.results_folder, 'Specifications.txt')
        with open(specs_file, 'w') as f:
            for key, val in self.specs.items():
                if key == 'kernel':
                    if self.multioutput_GP:
                        for k in range(self.model.nb_output_dims):
                            for i in range(len(
                                    self.model.models[k].kern.parameters)):
                                print(
                                    self.model.models[k].kern.parameters[i],
                                    file=f)
                    else:
                        for i in range(len(self.model.kern.parameters)):
                            print(self.model.kern.parameters[i], file=f)
                else:
                    print(key, ': ', val, file=f)

        # Plot computation time over time
        plt.close('all')
        name = 'Computation_time' + '.pdf'
        plt.plot(self.time[:, 0], self.time[:, 1], 'lime',
                 label='Time (s)')
        plt.title('Computation times over time')
        plt.legend()
        plt.xlabel('Number of samples')
        plt.ylabel('Computation time')
        plt.savefig(os.path.join(results_folder, name),
                    bbox_inches='tight')
        plt.close('all')

        # Plot RMSE over time
        if self.monitor_experiment:
            name = 'RMSE' + '.pdf'
            # self.RMSE_time = self.RMSE_time[
            #     np.logical_not(self.RMSE_time[:, 0] == 0)]
            plt.plot(self.RMSE_time[:, 0], self.RMSE_time[:, 1], 'c',
                     label='RMSE')
            plt.title('RMSE between model and true dynamics over time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('RMSE')
            plt.savefig(os.path.join(results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'SRMSE' + '.pdf'
            # self.SRMSE_time = self.SRMSE_time[
            #     np.logical_not(self.SRMSE_time[:, 0] == 0)]
            plt.plot(self.SRMSE_time[:, 0], self.SRMSE_time[:, 1], 'c',
                     label='SRMSE')
            plt.title('Standardized RMSE between model and true dynamics over '
                      'time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('SRMSE')
            plt.savefig(os.path.join(results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'Average_log_likelihood' + '.pdf'
            # self.log_likelihood_time = self.log_likelihood_time[
            #     np.logical_not(self.log_likelihood_time[:, 0] == 0)]
            plt.plot(self.log_likelihood_time[:, 0],
                     self.log_likelihood_time[:, 1], 'c', label='log_AL')
            plt.title(
                'Average log likelihood between model and true dynamics over '
                'time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Average log likelihood')
            plt.savefig(os.path.join(results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

            name = 'Standardized_average_log_likelihood' + '.pdf'
            # self.stand_log_likelihood_time = self.stand_log_likelihood_time[
            #     np.logical_not(self.stand_log_likelihood_time[:, 0] == 0)]
            plt.plot(self.stand_log_likelihood_time[:, 0],
                     self.stand_log_likelihood_time[:, 1], 'c', label='log_AL')
            plt.title(
                'Standardized average log likelihood between model and true '
                'dynamics over  time')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Standardized average log likelihood')
            plt.savefig(os.path.join(results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

        # Plot histogram of L2 error over grid
        l2_error_array = self.variables['l2_error_array']
        for i in range(self.grid.shape[1]):
            for j in range(self.Y.shape[1]):
                name = 'Histogram2d_L2_error' + str(i) + str(j) + '.pdf'
                nb_bins = 40
                plt.hist2d(self.grid[:, i], l2_error_array[:, j], bins=nb_bins)
                plt.title('Repartition of L2 errors over evaluation grid')
                plt.xlabel('Evaluation point ' + str(i))
                plt.ylabel('L2 error' + str(j))
                plt.savefig(os.path.join(results_folder, name),
                            bbox_inches='tight')
                plt.close('all')
        filename = 'l2_error_array.csv'
        file = pd.DataFrame(self.variables['l2_error_array'])
        file.to_csv(os.path.join(self.results_folder, filename),
                    header=False)

        # Plot evolution of hyperparameters over time
        for i in range(1, self.hyperparams.shape[1]):
            name = 'Hyperparameter' + str(i) + '.pdf'
            plt.plot(self.hyperparams[:, 0], self.hyperparams[:, i],
                     c='darkblue', label='Hyperparameter')
            plt.title('Evolution of hyperparameters during exploration')
            plt.legend()
            plt.xlabel('Number of samples')
            plt.ylabel('Hyperparameter value')
            plt.savefig(os.path.join(results_folder, name),
                        bbox_inches='tight')
            plt.close('all')

        logging.info(self.results_folder)

    def save(self):
        # Retrieve all complete variables and save everything in folder
        l2_error, RMSE, SRMSE, self.grid_variables['Predicted_grid'], \
        self.grid_variables['True_predicted_grid'], _, log_likelihood, \
        stand_log_likelihood = self.evaluate_model()
        self.RMSE_time = np.concatenate((self.RMSE_time, reshape_pt1(
            np.array([self.sample_idx, RMSE]))), axis=0)
        self.SRMSE_time = np.concatenate((self.SRMSE_time, reshape_pt1(
            np.array([self.sample_idx, SRMSE]))), axis=0)
        self.log_likelihood_time = np.concatenate(
            (self.log_likelihood_time, reshape_pt1(
                np.array([self.sample_idx, log_likelihood]))), axis=0)
        self.stand_log_likelihood_time = np.concatenate(
            (self.stand_log_likelihood_time, reshape_pt1(
                np.array([self.sample_idx, stand_log_likelihood]))), axis=0)
        self.variables['RMSE_time'] = self.RMSE_time
        self.variables['SRMSE_time'] = self.SRMSE_time
        self.variables['log_likelihood_time'] = self.log_likelihood_time
        self.variables[
            'stand_log_likelihood_time'] = self.stand_log_likelihood_time
        self.specs['l2_error'] = l2_error
        self.specs['RMSE'] = RMSE
        self.specs['SRMSE'] = SRMSE
        self.specs['log_AL'] = log_likelihood
        self.specs['stand_log_AL'] = stand_log_likelihood
        self.specs['X_mean'] = self.scaler_X.mean_
        self.specs['X_var'] = self.scaler_X.var_
        self.specs['Y_mean'] = self.scaler_Y.mean_
        self.specs['Y_var'] = self.scaler_Y.var_
        for key, val in self.grid_variables.items():
            if key == 'Predicted_grid':
                filename = str(key) + '.csv'
                file = pd.DataFrame(val)
                file.to_csv(os.path.join(self.results_folder, filename),
                            header=False)

        filename = 'Hyperparameters.csv'
        file = pd.DataFrame(self.hyperparams)
        file.to_csv(os.path.join(self.results_folder, filename),
                    header=False)

        if self.step > 1 and self.memory_saving:
            self.save_intermediate(memory_saving=True, ignore_first=True)
            # Read complete variables at the end
            self.read_grid_variables(self.results_folder)
            self.read_variables(self.results_folder)
            self.X = self.variables['X']
            self.Y = self.variables['Y']
            self.U = self.variables['U']
            self.time = self.variables['Computation_time']
            if self.monitor_experiment:
                self.RMSE_time = self.variables['RMSE_time']
                self.SRMSE_time = self.variables['SRMSE_time']
                self.log_likelihood_time = self.variables['log_AL_time']
                self.stand_log_likelihood_time = self.variables[
                    'stand_log_AL_time']
        else:
            self.save_intermediate(memory_saving=False)

        # Evaluate 1 step ahead prediction error over grid
        model_evaluation(self.grid_variables['Evaluation_grid'],
                         self.grid_variables['Grid_controls'],
                         self.grid_variables['Predicted_grid'],
                         self.grid_variables['True_predicted_grid'],
                         self.results_folder,
                         ground_truth_approx=self.ground_truth_approx,
                         verbose=False)
        plot_GP(self, grid=np.concatenate((self.grid_variables[
                                               'Evaluation_grid'],
                                           self.grid_variables[
                                               'Grid_controls']), axis=1),
                verbose=self.verbose)
        save_GP_data(self, verbose=self.verbose)

        # Run rollouts
        self.evaluate_rollouts()

        if self.verbose:
            logging.info('L2 error per state')
            logging.info(self.specs['l2_error'])
            logging.info('Total RMSE (mean over grid)')
            logging.info(self.specs['RMSE'])
            logging.info('Total SRMSE (mean over grid)')
            logging.info(self.specs['SRMSE'])
            logging.info('Total log_AL (mean over grid)')
            logging.info(self.specs['log_AL'])
            logging.info('Total stand_log_AL (mean over grid)')
            logging.info(self.specs['stand_log_AL'])
            logging.info(
                'Rollout RMSE (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_RMSE'])
            logging.info(
                'Rollout SRMSE (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_SRMSE'])
            logging.info(
                'Rollout log_AL (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_stand_log_AL'])
            logging.info(
                'Rollout stand_log_AL (mean over ' + str(self.nb_rollouts) +
                ' rollouts)')
            logging.info(self.specs['rollout_stand_log_AL'])
        self.save_folder(self.results_folder)

        for key, val in self.variables.items():
            if key.startswith('test_') or key.startswith('val_') or (
                    'rollout' in key):
                # Avoid saving all test and validation variables, and only
                # save rollout variables in rollout functions
                continue
            filename = str(key) + '.csv'
            # Re-index all files so indexes all follow properly
            file = pd.read_csv(os.path.join(self.results_folder, filename),
                               sep=',', header=None)
            file = file.drop(file.columns[0], axis=1)
            file.reset_index(drop=True)
            file.to_csv(os.path.join(self.results_folder, filename),
                        mode='w', header=False)

        if self.verbose:
            logging.info('Final model:')
            logging.info(self.model)
            logging.info('Saved results in')
            logging.info(self.results_folder)
