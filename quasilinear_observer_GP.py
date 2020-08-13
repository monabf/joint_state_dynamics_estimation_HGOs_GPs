import logging
import os
import shutil
import sys

import GPy
import numpy as np
import pandas as pd
import seaborn as sb

from config import Config
from controllers import sin_controller_02D
from dynamics import dynamics_traj, duffing_dynamics, pendulum_dynamics, \
    VanderPol_dynamics, duffing_dynamics_discrete, \
    harmonic_oscillator_dynamics, duffing_modified_cossquare
from gain_adaptation_laws import simple_score_adapt_highgain, \
    Praly_highgain_adaptation_law
from observers import duffing_observer_Delgado, \
    dim1_observe_data, duffing_observer_Delgado_GP, \
    duffing_observer_Delgado_discrete, duffing_observer_Delgado_GP_discrete, \
    harmonic_oscillator_observer_GP, duffing_observer_Michelangelo_GP, \
    WDC_justvelocity_discrete_observer_highgain_GP, \
    WDC_justvelocity_observer_highgain_GP, \
    WDC_justvelocity_observer_adaptive_highgain_GP
from plotting_functions import save_outside_data, plot_outside_data
from prior_means import duffing_continuous_prior_mean, \
    duffing_discrete_prior_mean, duffing_continuous_to_discrete_prior_mean, \
    duffing_continuous_prior_mean_Michelangelo_u, \
    duffing_continuous_prior_mean_Michelangelo_deriv_u, \
    pendulum_continuous_prior_mean_Michelangelo_u, \
    pendulum_continuous_prior_mean_Michelangelo_deriv_u, \
    harmonic_oscillator_continuous_prior_mean, \
    harmonic_oscillator_continuous_to_discrete_prior_mean, \
    harmonic_oscillator_continuous_prior_mean_Michelangelo_u, \
    harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv, \
    harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv_u, \
    duffing_cossquare_continuous_prior_mean_Michelangelo_deriv, \
    duffing_cossquare_continuous_prior_mean_Michelangelo_deriv_u, \
    duffing_cossquare_continuous_prior_mean_Michelangelo_u, \
    VanderPol_continuous_prior_mean_Michelangelo_u, \
    VanderPol_continuous_prior_mean_Michelangelo_deriv, \
    VanderPol_continuous_prior_mean_Michelangelo_deriv_u, \
    wdc_arm_continuous_to_discrete_justvelocity_prior_mean, \
    wdc_arm_continuous_justvelocity_prior_mean
from simple_GP_dyn import Simple_GP_Dyn
from simulation_functions import simulate_dynamics, simulate_estimations, \
    form_GP_data
from utils import reshape_pt1, reshape_dim1, interpolate, reshape_dim1_tonormal

sb.set_style('whitegrid')

# Script to test quasi-linear system with observer, adding GP to learn
# nonlinear part

# Logging
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(
            '../Figures/Logs', 'log' + str(sys.argv[1]))),
        logging.StreamHandler(sys.stdout)
    ])


def start_log():
    logging.INFO
    logging.FileHandler("{0}/{1}.log".format(
        '../Figures/Logs', 'log' + str(sys.argv[1])))
    logging.StreamHandler(sys.stdout)


def stop_log():
    logging._handlers.clear()
    logging.shutdown()


def update_params_on_loop(system, dyn_kwargs):
    if 'Duffing' in system:
        # omega = np.random.uniform(0, 2 * np.pi, 1)
        # dyn_kwargs['omega'] = omega
        gamma = np.random.uniform(0.2, 0.9, 1)
        dyn_kwargs['gamma'] = gamma
    elif 'Pendulum' in system:
        omega = np.random.uniform(1, np.pi, 1)
        dyn_kwargs['omega'] = omega
        gamma = np.random.uniform(1, 5, 1)
        dyn_kwargs['gamma'] = gamma
    else:
        logging.warning('No parameter update defined for this system')
    return dyn_kwargs


if __name__ == '__main__':
    start_log()

    # General params
    true_meas_noise_var = 1e-5
    process_noise_var = 0
    system = 'Continuous/Duffing/Discrete_model/' \
             'GP_justvelocity_adaptive_highgain_observer_noisy_inputs'
    optim_method = 'RK45'
    nb_samples = 500
    t0_span = 0
    tf_span = 30
    t0 = 0
    tf = 30
    t_span = [t0_span, tf_span]
    t_eval = np.linspace(t0, tf, nb_samples)
    dt = (tf - t0) / nb_samples
    nb_rollouts = 10  # Must be 0 if not simple dyns GP or def predict_euler
    rollout_length = 300
    rollout_controller = {'random': 3, 'sin_controller_02D': 4,
                          'null_controller': 3}
    nb_loops = 10
    sliding_window_size = 3000
    verbose = False
    monitor_experiment = True
    multioutput_GP = False
    sparse = None
    memory_saving = False  # Frees up RAM but slows down
    restart_on_loop = False
    if t0 != 0 or not restart_on_loop:
        logging.warning(
            'Initial simulation time is not 0 for each scenario! This is '
            'incompatible with DynaROM.')
    GP_optim_method = 'lbfgsb'  # Default: 'lbfgsb'
    meas_noise_var = 0.1  # Large to account for state estimation errors
    hyperparam_optim = 'fixed_hyperparameters'  # For hyperparameter optim
    batch_adaptive_gain = None  # For gain adaptation
    assert not (batch_adaptive_gain and ('adaptive' in system)), \
        'Cannot adapt the gain both through a continuous dynamic and a ' \
        'batch adaptation law.'
    observer_prior_mean = None
    dyn_GP_prior_mean = None
    dyn_GP_prior_mean_deriv = None
    if 'Continuous_model' in system:
        continuous_model = True
    else:
        continuous_model = False

    # System params
    if 'Continuous/Duffing' in system:
        discrete = False
        dyn_GP_prior_mean_deriv = None
        dyn_kwargs = {'alpha': -1, 'beta': 1, 'delta': 0.3, 'gamma': 0.4,
                      'omega': 1.2, 'dt': dt, 'dt_before_subsampling': 0.001}
        dynamics = duffing_dynamics
        controller = sin_controller_02D
        init_state = reshape_pt1(np.array([[0, 1]]))
        init_state_estim = reshape_pt1(np.array([[0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        observe_data = dim1_observe_data
        if 'GP_Delgado' in system:
            observer = duffing_observer_Delgado_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -1, 'beta': 0.9,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            observer_prior_mean = duffing_continuous_prior_mean
            dyn_GP_prior_mean = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': 0, 'beta': 0,
                                          'delta': 0, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array([-30, -1])
            observer_prior_mean = None
            dyn_GP_prior_mean = \
                duffing_continuous_prior_mean_Michelangelo_u
            dyn_GP_prior_mean_deriv = \
                duffing_continuous_prior_mean_Michelangelo_deriv_u
            init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        elif 'GP_justvelocity_highgain' in system:
            observer = WDC_justvelocity_observer_highgain_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
                                                            'k2': 5}
            observer_prior_mean = None
            if continuous_model:
                dyn_GP_prior_mean = wdc_arm_continuous_justvelocity_prior_mean
                dyn_kwargs['saturation'] = np.array([30])
            else:
                dyn_GP_prior_mean = \
                    wdc_arm_continuous_to_discrete_justvelocity_prior_mean
                dyn_kwargs['saturation'] = np.array(
                    [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            dyn_GP_prior_mean_deriv = None
        elif 'GP_justvelocity_adaptive_highgain' in system:
            observer = WDC_justvelocity_observer_adaptive_highgain_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = \
                {'g': 15, 'k1': 5, 'k2': 5, 'p1': 300, 'p2': 1e-5,
                 'b': 1e-4, 'n': init_state.shape[1], 'adaptation_law':
                     Praly_highgain_adaptation_law}
            dyn_kwargs['saturation'] = np.array(
                [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            if continuous_model:
                dyn_GP_prior_mean = None
            else:
                dyn_GP_prior_mean = \
                    wdc_arm_continuous_to_discrete_justvelocity_prior_mean
            dyn_GP_prior_mean_deriv = None
            init_state_estim = reshape_pt1(np.array([[0, 0, dyn_kwargs[
                'prior_kwargs']['observer_gains']['g']]]))
        elif 'Delgado' in system:
            observer = duffing_observer_Delgado
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        elif 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -2
        grid_sup = 2
        # Create kernel
        if dyn_kwargs.get('gamma') == 0:
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=110,
                              lengthscale=np.array([5, 15, 150, 150]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(110, 10))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([5, 15, 150, 150]),
                                            np.diag([0.5, 1, 10, 10])))
    elif 'Discrete/Duffing' in system:
        discrete = True
        dyn_kwargs = {'alpha': -1, 'beta': 1, 'delta': 0.3, 'gamma': 0.4,
                      'omega': 1.2}
        dynamics = duffing_dynamics_discrete
        controller = sin_controller_02D
        if 'GP_Delgado' in system:
            observer = duffing_observer_Delgado_GP_discrete
            dyn_kwargs['prior_kwargs'] = {'alpha': -1, 'beta': 0.95,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            observer_prior_mean = duffing_discrete_prior_mean
            dyn_GP_prior_mean = duffing_continuous_to_discrete_prior_mean
        elif 'Delgado' in system:
            observer = duffing_observer_Delgado_discrete
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        elif 'GP_justvelocity_highgain_discrete' in system:
            observer = WDC_justvelocity_discrete_observer_highgain_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 15, 'k1': 5,
                                                            'k2': 5}
            dyn_kwargs['saturation'] = np.array(
                [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = \
                wdc_arm_continuous_to_discrete_justvelocity_prior_mean
            dyn_GP_prior_mean_deriv = None
        elif 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 1]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -5
        grid_sup = 5
        # Create kernel
        if dyn_kwargs.get('gamma') == 0:
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=47,
                              lengthscale=np.array([1, 1, 1, 1]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(30, 50))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([10, 10, 10, 10]),
                                            np.diag([50, 50, 50, 50])))
    elif 'Continuous/Pendulum' in system:
        discrete = False
        # dyn_kwargs = {'k': 0.05, 'm': 0.1, 'g': 9.8, 'l': 1, 'gamma': 5,
        #               'f0': 0.5, 'f1': 1 / (2 * np.pi), 't1': tf * nb_loops}
        # dynamics = pendulum_dynamics
        # controller = chirp_controller
        dyn_kwargs = {'k': 0.05, 'm': 0.1, 'g': 9.8, 'l': 1, 'gamma': 5.,
                      'omega': 1.2}
        dynamics = pendulum_dynamics
        controller = sin_controller_02D
        if 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0, 'm': 0.1, 'g': 0,
                                          'l': 1, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 20, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array([-5, 5])
            observer_prior_mean = None
            dyn_GP_prior_mean = pendulum_continuous_prior_mean_Michelangelo_u
            dyn_GP_prior_mean_deriv = \
                pendulum_continuous_prior_mean_Michelangelo_deriv_u
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 0]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -3
        grid_sup = 3
        # Create kernel
        if (dyn_kwargs.get('gamma') == 0) or (dyn_kwargs.get('gain') == 0):
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=60,
                              lengthscale=np.array([12, 18, 150, 150]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(60, 10))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([12, 18, 150, 150]),
                                            np.diag([5, 5, 50, 50])))
        meas_noise_var = 5e-3
    elif 'Continuous/Harmonic_oscillator' in system:
        discrete = False
        dyn_kwargs = {'k': 0.05, 'm': 0.05, 'gamma': 0, 'omega': 1.2}
        dynamics = harmonic_oscillator_dynamics
        controller = sin_controller_02D
        if 'GP_Luenberger_observer' in system:
            observer = harmonic_oscillator_observer_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0.048, 'm': 0.05, 'gamma': 0,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            observer_prior_mean = harmonic_oscillator_continuous_prior_mean
            dyn_GP_prior_mean = \
                harmonic_oscillator_continuous_to_discrete_prior_mean
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0.05, 'm': 0.05, 'gamma': 0,
                                          'omega': 1.2}
            dyn_kwargs['continuous_model'] = continuous_model
            observer_prior_mean = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo_u
            dyn_GP_prior_mean_deriv = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv_u
        elif 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[1, 0]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0])  # imposed instead u(t=0)!
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -2
        grid_sup = 2
        # Create kernel
        if dyn_kwargs.get('gamma') == 0:
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=47,
                              lengthscale=np.array([1, 1]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(30, 1))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([10, 10]),
                                            np.diag([1, 1])))
    elif 'Continuous/VanderPol' in system:
        discrete = False
        dyn_kwargs = {'mu': 2, 'gamma': 1.2, 'omega': np.pi / 10}
        dynamics = VanderPol_dynamics
        controller = sin_controller_02D
        if 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'mu': 2, 'gamma': 1.2,
                                          'omega': np.pi / 10, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 20, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [8 * dyn_kwargs.get('prior_kwargs').get('mu') - 1,
                 3 * dyn_kwargs.get('prior_kwargs').get('mu')])
            observer_prior_mean = \
                VanderPol_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = VanderPol_continuous_prior_mean_Michelangelo_u
            dyn_GP_prior_mean_deriv = \
                VanderPol_continuous_prior_mean_Michelangelo_deriv_u
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 4]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -2
        grid_sup = 2
        # Create kernel
        if dyn_kwargs.get('gamma') == 0:
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=30,
                              lengthscale=np.array([1, 3, 150, 150]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(30, 10))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([1, 3, 150, 150]),
                                            np.diag([1, 1, 50, 50])))
    elif 'Continuous/Modified_Duffing_Cossquare' in system:
        discrete = False
        dyn_GP_prior_mean_deriv = None
        dyn_kwargs = {'alpha': 2, 'beta': 2, 'delta': 0.3, 'gamma': 0.4,
                      'omega': 1.2}
        dynamics = duffing_modified_cossquare
        controller = sin_controller_02D
        if 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': 2, 'beta': 2,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['continuous_model'] = continuous_model
            dyn_kwargs['saturation'] = np.array(
                [- 5 * dyn_kwargs.get('beta') - 5 * dyn_kwargs.get('alpha'),
                 -5 * dyn_kwargs.get('delta')])
            observer_prior_mean = \
                duffing_cossquare_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = \
                duffing_cossquare_continuous_prior_mean_Michelangelo_u
            dyn_GP_prior_mean_deriv = \
                duffing_cossquare_continuous_prior_mean_Michelangelo_deriv_u
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 1]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -1
        grid_sup = 1
        # Create kernel
        if dyn_kwargs.get('gamma') == 0:
            input_dim = init_state.shape[1]
        else:
            input_dim = init_state.shape[1] + init_control.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=110,
                              lengthscale=np.array([5, 15, 150, 150]),
                              ARD=True)
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(110, 10))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([5, 15, 150, 150]),
                                            np.diag([0.1, 0.5, 10, 10])))
    else:
        raise Exception('Unknown system')

    # Set derivative_function for continuous models
    if continuous_model:
        if dyn_GP_prior_mean:
            logging.warning('A prior mean has been defined for the GP though '
                            'a continuous model is being used. Check this is '
                            'really what you want to do, as a prior mean is '
                            'often known for discrete models without being '
                            'available for continuous ones.')


        def derivative_function(X, U, y_observed, GP):
            X = reshape_pt1(X)
            u = lambda t, kwargs, t0, init_control: reshape_pt1(U)[t]
            y = lambda t, kwargs: reshape_pt1(y_observed)[t]
            Xdot = np.array([observer(t, X[t], u, y, t0, init_control, GP,
                                      dyn_kwargs) for t in
                             range(len(X))])
            Xdot = Xdot.reshape(X.shape)
            return Xdot.reshape(X.shape)
    else:
        derivative_function = None

    # Generate data: simulate dynamics
    xtraj, utraj, t_utraj = simulate_dynamics(t_span=t_span, t_eval=t_eval,
                                              t0=t0, dt=dt,
                                              init_control=init_control,
                                              init_state=init_state,
                                              dynamics=dynamics,
                                              controller=controller,
                                              process_noise_var=process_noise_var,
                                              optim_method=optim_method,
                                              dyn_config=dyn_kwargs,
                                              discrete=discrete,
                                              verbose=verbose)

    # Observe data: only position, observer reconstitutes velocity
    # Get observations over t_eval and simulate xhat only over t_eval
    y_observed, t_y, xtraj_estim = \
        simulate_estimations(system=system, observe_data=observe_data,
                             t_eval=t_eval, t0=t0, tf=tf, dt=dt,
                             meas_noise_var=true_meas_noise_var,
                             init_control=init_control,
                             init_state_estim=init_state_estim,
                             controller=controller, observer=observer,
                             optim_method=optim_method,
                             dyn_config=dyn_kwargs, xtraj=xtraj,
                             GP=observer_prior_mean, discrete=discrete,
                             verbose=verbose)

    # Create initial data for GP, noiseless or noisy X, noiseless U, noisy Y
    X, U, Y = form_GP_data(system=system, xtraj=xtraj,
                           xtraj_estim=xtraj_estim, utraj=utraj,
                           meas_noise_var=true_meas_noise_var,
                           y_observed=y_observed,
                           derivative_function=derivative_function,
                           model=observer_prior_mean)

    # True dynamics: (xt, ut) -> xt+1 if no observer, (xt, ut) -> phi(xt,ut) if
    # Michelangelo. If no observer, simulate system for 10*dt starting at xt
    # and return result at t+dt
    if ('Michelangelo' in system) and ('Duffing' in system):
        # Return xi_t instead of x_t+1 from x_t,u_t
        true_dynamics = lambda x, control: \
            - dyn_kwargs.get('beta') * x[:, 0] ** 3 - dyn_kwargs.get('alpha') \
            * x[:, 0] - dyn_kwargs.get('delta') * x[:, 1]
    elif ('justvelocity' in system) and ('Duffing' in system):
        if not continuous_model:
            true_dynamics = lambda x, control: dynamics_traj(
                x0=reshape_pt1(x), u=lambda t, kwarg, t0, init_control:
                interpolate(t, np.concatenate((reshape_dim1(np.arange(
                    len(control))), control), axis=1),
                            t0=t0, init_value=init_control),
                t0=t0, dt=dt, init_control=init_control, version=dynamics,
                meas_noise_var=0, process_noise_var=process_noise_var,
                method=optim_method, t_span=[0, dt], t_eval=[dt],
                kwargs=dyn_kwargs)[:, -1]
        else:
            true_dynamics = lambda x, control: \
                dynamics(t=t0, x=x, u=lambda t, kwarg, t0, init_control:
                interpolate(t, np.concatenate((reshape_dim1(np.arange(
                    len(control))), control), axis=1), t0=t0,
                            init_value=init_control),
                         t0=t0, init_control=control,
                         process_noise_var=process_noise_var,
                         kwargs=dyn_kwargs)[:, -1]
    elif ('Michelangelo' in system) and ('Harmonic_oscillator' in system):
        # Return xi_t instead of x_t+1 from x_t,u_t
        true_dynamics = lambda x, control: \
            - dyn_kwargs.get('k') / dyn_kwargs.get('m') * x[:, 0]
    elif ('Michelangelo' in system) and ('Pendulum' in system):
        # Return xi_t instead of x_t+1 from x_t,u_t
        true_dynamics = lambda x, control: \
            - dyn_kwargs.get('g') / dyn_kwargs.get('l') * np.sin(x[:, 0]) \
            - dyn_kwargs.get('k') / dyn_kwargs.get('m') * x[:, 1]
    elif ('Michelangelo' in system) and ('VanderPol' in system):
        # Return xi_t instead of x_t+1 from x_t,u_t
        true_dynamics = lambda x, control: reshape_pt1(
            dyn_kwargs.get('mu') * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0])
    elif (('Michelangelo' in system) or ('justvelocity_highgain' in system)) \
            and not any(k in system for k in ('Duffing', 'Harmonic_oscillator',
                                              'Pendulum', 'VanderPol')):
        raise Exception('No ground truth has been defined.')
    else:
        true_dynamics = lambda x, control: dynamics_traj(
            x0=reshape_pt1(x), u=lambda t, kwarg, t0, init_control:
            interpolate(t, np.concatenate((reshape_dim1(np.arange(
                len(control))), control), axis=1),
                        t0=t0, init_value=init_control),
            t0=t0, dt=dt, init_control=init_control, version=dynamics,
            meas_noise_var=0, process_noise_var=process_noise_var,
            method=optim_method, t_span=[0, dt], t_eval=[dt],
            kwargs=dyn_kwargs)

    # Create config file from all params (not optimal, for cluster use
    # make cluster_this_script.py in which config is directly a system
    # argument given in command line and chosen from a set of predefined
    # config files)
    if not controller or not np.any(utraj):
        no_control = True
    else:
        no_control = False
    config = Config(true_meas_noise_var=true_meas_noise_var,
                    process_noise_var=process_noise_var,
                    system=system,
                    optim_method=optim_method,
                    nb_samples=nb_samples,
                    t0_span=t0_span,
                    tf_span=tf_span,
                    t0=t0,
                    tf=tf,
                    dt=dt,
                    dt_before_subsampling=dyn_kwargs['prior_kwargs'][
                        'dt_before_subsampling'],
                    nb_rollouts=nb_rollouts,
                    rollout_length=rollout_length,
                    rollout_controller=rollout_controller,
                    nb_loops=nb_loops,
                    sliding_window_size=sliding_window_size,
                    verbose=verbose,
                    monitor_experiment=monitor_experiment,
                    multioutput_GP=multioutput_GP,
                    sparse=sparse,
                    memory_saving=memory_saving,
                    restart_on_loop=restart_on_loop,
                    GP_optim_method=GP_optim_method,
                    meas_noise_var=meas_noise_var,
                    hyperparam_optim=hyperparam_optim,
                    batch_adaptive_gain=batch_adaptive_gain,
                    discrete=discrete,
                    dynamics=dynamics,
                    controller=controller,
                    init_state=init_state,
                    init_state_estim=init_state_estim,
                    init_control=init_control,
                    input_dim=input_dim,
                    observer=observer,
                    true_dynamics=true_dynamics,
                    no_control=no_control,
                    dyn_kwargs=dyn_kwargs,
                    prior_kwargs=dyn_kwargs['prior_kwargs'],
                    observer_gains=dyn_kwargs['prior_kwargs'][
                        'observer_gains'],
                    saturation=dyn_kwargs['saturation'],
                    observer_prior_mean=observer_prior_mean,
                    prior_mean=dyn_GP_prior_mean,
                    prior_mean_deriv=dyn_GP_prior_mean_deriv,
                    derivative_function=derivative_function,
                    continuous_model=continuous_model,
                    observe_data=observe_data,
                    constrain_u=constrain_u,
                    constrain_x=constrain_x,
                    grid_inf=grid_inf,
                    grid_sup=grid_sup,
                    kernel=kernel)
    config.update(dyn_kwargs)
    config.dyn_kwargs.update(saturation=config.saturation,
                             prior_kwargs=config.prior_kwargs)
    config.dyn_kwargs['prior_kwargs']['observer_gains'] = config.observer_gains

    # Create GP
    dyn_kwargs.update({'dt': dt, 't0': t0, 'tf': tf, 't_span': t_span,
                       'init_state': init_state,
                       'init_state_estim': init_state_estim,
                       'init_control': init_control,
                       'observer_prior_mean': observer_prior_mean,
                       'true_noise_var': true_meas_noise_var,
                       'batch_adaptive_gain': batch_adaptive_gain})
    dyn_GP = Simple_GP_Dyn(X, U, Y, config)

    # Learn simple GP of dynamics, by seeing pairs (x_t, u_t) -> y_t
    data_to_save = {'xtraj': xtraj, 'xtraj_estim': xtraj_estim,
                    'y_observed': y_observed}
    if batch_adaptive_gain:
        gain_time = np.array(
            [dyn_kwargs['prior_kwargs']['observer_gains']['g']])
        data_to_save.update({'gain_time': gain_time})
    elif 'adaptive' in system:
        output_error = reshape_dim1(np.square(xtraj[:, 0] - xtraj_estim[:, 0]))
        gain_time = reshape_dim1(xtraj_estim[:, -1])
        data_to_save.update(
            {'gain_time': gain_time, 'output_error': output_error})
    save_outside_data(dyn_GP, data_to_save)
    plot_outside_data(dyn_GP, data_to_save)
    dyn_GP.learn()

    # Run rollouts using only priors, before learning (step=-1)
    rollouts_folder = os.path.join(dyn_GP.results_folder, 'Rollouts_0')
    new_rollouts_folder = os.path.join(dyn_GP.results_folder,
                                       'Rollouts_-1')
    shutil.copytree(rollouts_folder, new_rollouts_folder)
    old_step, dyn_GP.step = dyn_GP.step, 0
    old_sample_idx, dyn_GP.sample_idx = dyn_GP.sample_idx, 0
    if 'justvelocity_adaptive' in config.system:
        # Do not adapt observer gains for closed-loop rollouts
        dyn_GP.evaluate_closedloop_rollouts(
            WDC_justvelocity_observer_highgain_GP,
            config.observe_data, no_GP_in_observer=True)
        if config.prior_mean:
            dyn_GP.evaluate_kalman_rollouts(
                WDC_justvelocity_observer_highgain_GP,
                config.observe_data, config.discrete,
                no_GP_in_observer=True, only_prior=True)
    else:
        dyn_GP.evaluate_closedloop_rollouts(
            config.observer, config.observe_data,
            no_GP_in_observer=True)
        if config.prior_mean:
            dyn_GP.evaluate_kalman_rollouts(
                config.observer, config.observe_data, config.discrete,
                no_GP_in_observer=True, only_prior=True)
    if config.prior_mean:
        # Also run open-loop rollouts with prior before learning
        dyn_GP.evaluate_rollouts(only_prior=True)
    dyn_GP.step = old_step
    dyn_GP.sample_idx = old_sample_idx

    dyn_GP.save()
    if 'justvelocity_adaptive' in system:
        # Do not adapt observer gains for closed-loop rollouts
        dyn_GP.evaluate_kalman_rollouts(
            WDC_justvelocity_observer_highgain_GP, observe_data, discrete)
        dyn_GP.evaluate_closedloop_rollouts(
            WDC_justvelocity_observer_highgain_GP, observe_data)
    else:
        dyn_GP.evaluate_kalman_rollouts(observer, observe_data, discrete)
        dyn_GP.evaluate_closedloop_rollouts(observer, observe_data)

    # Alternate between estimating xtraj from observations (or just getting
    # new xtraj), estimating fhat from new xtraj(_estim), and loop
    for i in range(1, nb_loops):
        # Update params and initial states after the first pass
        if restart_on_loop:
            dyn_kwargs = update_params_on_loop(system, dyn_kwargs)
        else:
            init_state = reshape_pt1(xtraj[-1])
            init_control = reshape_pt1(utraj[-1])
            init_state_estim = reshape_pt1(xtraj_estim[-1])
            tf_before = tf
            tf_span = tf_before + (tf_span - t0_span)
            t0_span = tf_before
            tf = tf_before + (tf - t0)
            t0 = tf_before
            t_span = [t0_span, tf_span]
        t_eval = np.linspace(t0, tf, nb_samples)
        dt = (tf - t0) / nb_samples

        # Update observer gain
        if batch_adaptive_gain:
            if 'simple_score_posdist_lastbatch' in batch_adaptive_gain:
                gain = dyn_kwargs['prior_kwargs']['observer_gains'].get('g')
                score = np.linalg.norm(
                    reshape_dim1_tonormal(xtraj_estim[:, 0] - xtraj[:, 0]))
                previous_idx = int(np.min([i, 2]))
                (base_path, loop) = os.path.split(dyn_GP.results_folder)
                previous_results_folder = os.path.join(
                    base_path, 'Loop_' + str(i - previous_idx))
                previous_xtraj_estim = pd.read_csv(os.path.join(
                    previous_results_folder, 'Data_outside_GP/xtraj_estim.csv'),
                    sep=',', header=None)
                previous_xtraj_estim = previous_xtraj_estim.drop(
                    previous_xtraj_estim.columns[0], axis=1).values
                previous_xtraj = pd.read_csv(os.path.join(
                    previous_results_folder, 'Data_outside_GP/xtraj.csv'),
                    sep=',', header=None)
                previous_xtraj = previous_xtraj.drop(
                    previous_xtraj.columns[0], axis=1).values
                previous_score = np.linalg.norm(reshape_dim1_tonormal(
                    previous_xtraj_estim[:, 0] - previous_xtraj[:, 0]))
                new_gain = simple_score_adapt_highgain(gain, score,
                                                       previous_score)
                dyn_kwargs['prior_kwargs']['observer_gains']['g'] = new_gain
                gain_time = np.concatenate((gain_time, np.array([new_gain])))
            elif 'simple_score_' in batch_adaptive_gain:
                param = batch_adaptive_gain.split('simple_score_', 1)[1]
                gain = dyn_kwargs['prior_kwargs']['observer_gains'].get('g')
                score = dyn_GP.variables[param][-1, 1]
                previous_idx = int(np.min([i, 2]))
                previous_score = dyn_GP.variables[param][-previous_idx, 1]
                new_gain = simple_score_adapt_highgain(gain, score,
                                                       previous_score)
                dyn_kwargs['prior_kwargs']['observer_gains']['g'] = new_gain
                gain_time = np.concatenate((gain_time, np.array([new_gain])))
            elif batch_adaptive_gain == 'change_last_batch':
                if i == nb_loops - 1:
                    new_gain = 3
                    dyn_kwargs['prior_kwargs']['observer_gains']['g'] = new_gain
                    gain_time = np.concatenate(
                        (gain_time, np.array([new_gain])))
            else:
                logging.error('This adaptation law for the observer gains has '
                              'not been defined.')

        (base_path, loop) = os.path.split(dyn_GP.results_folder)
        new_results_folder = os.path.join(base_path, 'Loop_' + str(i))
        os.makedirs(new_results_folder, exist_ok=False)
        dyn_GP.set_results_folder(new_results_folder)
        dyn_GP.set_dyn_kwargs(dyn_kwargs)

        # Create new data, by simulating again starting from the newt
        # init_state and init_state_estim, and re-learn GP
        xtraj, utraj, t_utraj = simulate_dynamics(t_span=t_span,
                                                  t_eval=t_eval,
                                                  t0=t0, dt=dt,
                                                  init_control=init_control,
                                                  init_state=init_state,
                                                  dynamics=dynamics,
                                                  controller=controller,
                                                  process_noise_var=process_noise_var,
                                                  optim_method=optim_method,
                                                  dyn_config=dyn_kwargs,
                                                  discrete=discrete,
                                                  verbose=verbose)
        if observer:
            y_observed, t_y, xtraj_estim = \
                simulate_estimations(system=system, observe_data=observe_data,
                                     t_eval=t_eval, t0=t0, tf=tf, dt=dt,
                                     meas_noise_var=true_meas_noise_var,
                                     init_control=init_control,
                                     init_state_estim=init_state_estim,
                                     controller=controller, observer=observer,
                                     optim_method=optim_method,
                                     dyn_config=dyn_kwargs, xtraj=xtraj,
                                     GP=dyn_GP, discrete=discrete,
                                     verbose=verbose)
        else:
            logging.info('No observer has been specified, using true data for '
                         'learning.')
            xtraj_estim = xtraj
        X, U, Y = form_GP_data(system=system, xtraj=xtraj,
                               xtraj_estim=xtraj_estim, utraj=utraj,
                               meas_noise_var=true_meas_noise_var,
                               y_observed=y_observed,
                               derivative_function=derivative_function,
                               model=dyn_GP)

        data_to_save = {'xtraj': xtraj, 'xtraj_estim': xtraj_estim,
                        'y_observed': y_observed}
        if batch_adaptive_gain:
            data_to_save.update({'gain_time': gain_time})
        elif 'adaptive' in system:
            output_error = reshape_dim1(np.square(
                xtraj[:, 0] - xtraj_estim[:, 0]))
            gain_time = np.concatenate((
                gain_time, reshape_dim1(xtraj_estim[:, -1])))
            data_to_save.update(
                {'gain_time': gain_time, 'output_error': output_error})
        save_outside_data(dyn_GP, data_to_save)
        plot_outside_data(dyn_GP, data_to_save)
        dyn_GP.learn(new_X=X, new_Y=Y, new_U=U)
        dyn_GP.save()
        if 'justvelocity_adaptive' in system:
            # Do not adapt observer gains for closed-loop rollouts
            dyn_GP.evaluate_kalman_rollouts(
                WDC_justvelocity_observer_highgain_GP, observe_data, discrete)
            dyn_GP.evaluate_closedloop_rollouts(
                WDC_justvelocity_observer_highgain_GP, observe_data)
        else:
            dyn_GP.evaluate_kalman_rollouts(observer, observe_data, discrete)
            dyn_GP.evaluate_closedloop_rollouts(observer, observe_data)

    stop_log()
