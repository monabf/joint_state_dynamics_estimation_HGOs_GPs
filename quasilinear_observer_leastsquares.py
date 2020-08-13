import logging
import os
import sys

import GPy
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import linear_model

from controllers import sin_controller
from dynamics import duffing_dynamics, pendulum_dynamics, \
    VanderPol_dynamics, duffing_dynamics_discrete, \
    harmonic_oscillator_dynamics, duffing_modified_cossquare
from gain_adaptation_laws import Praly_highgain_adaptation_law
from observers import duffing_observer_Delgado, \
    dim1_observe_data, duffing_observer_Delgado_GP, \
    duffing_observer_Delgado_discrete, duffing_observer_Delgado_GP_discrete, \
    harmonic_oscillator_observer_GP, duffing_observer_Michelangelo_GP, \
    WDC_justvelocity_discrete_observer_highgain_GP, \
    WDC_justvelocity_observer_highgain_GP, \
    WDC_justvelocity_observer_adaptive_highgain_GP, \
    duffing_observer_Michelangelo_LS
from prior_means import duffing_continuous_prior_mean, \
    duffing_discrete_prior_mean, duffing_continuous_to_discrete_prior_mean, \
    harmonic_oscillator_continuous_prior_mean, \
    harmonic_oscillator_continuous_to_discrete_prior_mean, \
    harmonic_oscillator_continuous_prior_mean_Michelangelo, \
    harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv, \
    duffing_cossquare_continuous_prior_mean_Michelangelo_deriv, \
    duffing_cossquare_continuous_prior_mean_Michelangelo_deriv_u, \
    duffing_cossquare_continuous_prior_mean_Michelangelo, \
    VanderPol_continuous_prior_mean_Michelangelo, \
    VanderPol_continuous_prior_mean_Michelangelo_deriv, \
    VanderPol_continuous_prior_mean_Michelangelo_deriv_u, \
    wdc_arm_continuous_to_discrete_justvelocity_prior_mean
from simulation_functions import simulate_dynamics, simulate_estimations, \
    form_GP_data
from utils import reshape_pt1, reshape_dim1, \
    RMS

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
    meas_noise_var = 1e-5
    process_noise_var = 0
    system = 'Continuous/Duffing/' \
             'LS_justvelocity_highgain_observer_noisy_inputs'
    optim_method = 'RK45'
    nb_samples = 500
    t0_span = 0
    tf_span = 60
    t0 = 0
    tf = 30
    t_span = [t0_span, tf_span]
    t_eval = np.linspace(t0, tf, nb_samples)
    dt = (tf - t0) / nb_samples
    nb_rollouts = 10  # Must be 0 if not simple dyns GP or def predict_euler
    nb_loops = 50
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
    batch_adaptive_gain = None  # For gain adaptation
    assert not (batch_adaptive_gain and ('adaptive' in system)), \
        'Cannot adapt the gain both through a continuous dynamic and a ' \
        'batch adaptation law.'
    observer_prior_mean = None

    # System params
    if 'Continuous/Duffing' in system:
        discrete = False
        dyn_GP_prior_mean_deriv = None
        dyn_kwargs = {'alpha': -1, 'beta': 1, 'delta': 0.3, 'gamma': 0.4,
                      'omega': 1.2, 'dt': dt, 'dt_before_subsampling': 0.001}
        dynamics = duffing_dynamics
        controller = sin_controller
        init_state = reshape_pt1(np.array([[0, 1]]))
        init_state_estim = reshape_pt1(np.array([[0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        if 'GP_Delgado' in system:
            observer = duffing_observer_Delgado_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -1, 'beta': 0.9,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            observer_prior_mean = duffing_continuous_prior_mean
            dyn_GP_prior_mean = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [- 30 * dyn_kwargs.get('prior_kwargs').get('beta'),
                 -5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
            init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        elif 'LS_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_LS
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [- 30 * dyn_kwargs.get('prior_kwargs').get('beta'),
                 -5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
            init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        elif 'GP_justvelocity_highgain' in system:
            observer = WDC_justvelocity_observer_highgain_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
                                                            'k2': 5}
            dyn_kwargs['saturation'] = np.array(
                [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = \
                wdc_arm_continuous_to_discrete_justvelocity_prior_mean
            dyn_GP_prior_mean_deriv = None
        elif 'LS_justvelocity_highgain' in system:
            # observer = WDC_justvelocity_observer_highgain_LS
            # dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
            #                               'delta': 0.2, 'gamma': 0.4,
            #                               'omega': 1.2, 'dt': dt,
            #                               'dt_before_subsampling': 0.001}
            # dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 8, 'k1': 5,
            #                                                 'k2': 5}
            # dyn_kwargs['saturation'] = np.array(
            #     [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            # observer_prior_mean = None
            # dyn_GP_prior_mean = \
            #     wdc_arm_continuous_to_discrete_justvelocity_prior_mean
            # dyn_GP_prior_mean_deriv = None
            observer = duffing_observer_Michelangelo_LS
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 5, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [- 30 * dyn_kwargs.get('prior_kwargs').get('beta'),
                 -5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
            init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        elif 'GP_justvelocity_adaptive_highgain' in system:
            observer = WDC_justvelocity_observer_adaptive_highgain_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': -0.5, 'beta': 1.3,
                                          'delta': 0.2, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = \
                {'g': 15, 'k1': 5, 'k2': 5, 'p1': 600, 'p2': 1e-5,
                 'b': 1e-4, 'n': init_state.shape[1], 'adaptation_law':
                     Praly_highgain_adaptation_law}
            dyn_kwargs['saturation'] = np.array(
                [-5 * dyn_kwargs.get('prior_kwargs').get('delta')])
            observer_prior_mean = None
            dyn_GP_prior_mean = \
                wdc_arm_continuous_to_discrete_justvelocity_prior_mean
            dyn_GP_prior_mean_deriv = None
            init_state_estim = reshape_pt1(np.array([[0, 0, 15]]))
        elif 'Delgado' in system:
            observer = duffing_observer_Delgado
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        elif 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        observe_data = dim1_observe_data
        constrain_u = [-dyn_kwargs.get('gamma'),
                       dyn_kwargs.get('gamma')]  # must be a python list!
        constrain_x = []  # must be a python list!
        grid_inf = -3
        grid_sup = 3
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
    elif 'Discrete/Duffing' in system:
        discrete = True
        dyn_kwargs = {'alpha': -1, 'beta': 1, 'delta': 0.3, 'gamma': 0.4,
                      'omega': 1.2}
        dynamics = duffing_dynamics_discrete
        controller = sin_controller
        if 'GP_Delgado' in system:
            observer = duffing_observer_Delgado_GP_discrete
            dyn_kwargs['prior_kwargs'] = {'alpha': -1, 'beta': 0.95,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
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
        constrain_u = []  # must be a python list!
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
        controller = sin_controller
        if 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0.05, 'm': 0.1, 'g': 9.8,
                                          'l': 1, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 20, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [5 * dyn_kwargs.get('prior_kwargs').get('g') / dyn_kwargs.get(
                    'prior_kwargs').get('l'),
                 5 * dyn_kwargs.get('prior_kwargs').get('k') / dyn_kwargs.get(
                     'prior_kwargs').get('m')])
            observer_prior_mean = None
            dyn_GP_prior_mean = None
            dyn_GP_prior_mean_deriv = None
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 0.5]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = []  # must be a python list!
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
        GP_meas_noise_var = 5e-3
    elif 'Continuous/Harmonic_oscillator' in system:
        discrete = False
        dyn_kwargs = {'k': 0.05, 'm': 0.05, 'gamma': 0, 'omega': 1.2}
        dynamics = harmonic_oscillator_dynamics
        controller = sin_controller
        if 'GP_Luenberger_observer' in system:
            observer = harmonic_oscillator_observer_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0.048, 'm': 0.05, 'gamma': 0,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            observer_prior_mean = harmonic_oscillator_continuous_prior_mean
            dyn_GP_prior_mean = \
                harmonic_oscillator_continuous_to_discrete_prior_mean
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'k': 0.05, 'm': 0.05, 'gamma': 0,
                                          'omega': 1.2}
            observer_prior_mean = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo
            dyn_GP_prior_mean_deriv = \
                harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv
        elif 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[1, 0]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0])  # imposed instead u(t=0)!
        constrain_u = []  # must be a python list!
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
        controller = sin_controller
        if 'No_observer' in system:
            observer = None
            observer_prior_mean = None
            dyn_GP_prior_mean = None
        elif 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'mu': 2, 'gamma': 1.2,
                                          'omega': np.pi / 10, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['prior_kwargs']['observer_gains'] = {'g': 20, 'k1': 5,
                                                            'k2': 5, 'k3': 1}
            dyn_kwargs['saturation'] = np.array(
                [8 * dyn_kwargs.get('prior_kwargs').get('mu') - 1,
                 3 * dyn_kwargs.get('prior_kwargs').get('mu')])
            observer_prior_mean = \
                VanderPol_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = VanderPol_continuous_prior_mean_Michelangelo
            dyn_GP_prior_mean_deriv = \
                VanderPol_continuous_prior_mean_Michelangelo_deriv_u
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 4]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = []  # must be a python list!
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
        controller = sin_controller
        if 'GP_Michelangelo' in system:
            observer = duffing_observer_Michelangelo_GP
            dyn_kwargs['prior_kwargs'] = {'alpha': 2, 'beta': 2,
                                          'delta': 0.3, 'gamma': 0.4,
                                          'omega': 1.2, 'dt': dt,
                                          'dt_before_subsampling': 0.001}
            dyn_kwargs['saturation'] = np.array(
                [- 5 * dyn_kwargs.get('beta') - 5 * dyn_kwargs.get('alpha'),
                 -5 * dyn_kwargs.get('delta')])
            observer_prior_mean = \
                duffing_cossquare_continuous_prior_mean_Michelangelo_deriv
            dyn_GP_prior_mean = \
                duffing_cossquare_continuous_prior_mean_Michelangelo
            dyn_GP_prior_mean_deriv = \
                duffing_cossquare_continuous_prior_mean_Michelangelo_deriv_u
        observe_data = dim1_observe_data
        init_state = reshape_pt1(np.array([[0, 1]]))
        init_state_estim = reshape_pt1(np.array([[0, 0, 0]]))
        init_control = reshape_pt1([0, 0])  # imposed instead u(t=0)!
        constrain_u = []  # must be a python list!
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

    # Generate data: simulate dynamics
    xtraj, utraj, t_utraj = simulate_dynamics(t_span=t_span, t_eval=t_eval,
                                              t0=t0, dt=dt,
                                              init_control=init_control,
                                              init_state=init_state,
                                              dynamics=dynamics,
                                              controller=controller,
                                              process_noise_var=process_noise_var,
                                              optim_method=optim_method,
                                              dyn_kwargs=dyn_kwargs,
                                              discrete=discrete,
                                              verbose=verbose)

    # Observe data: only position, observer reconstitutes velocity
    # Get observations over t_eval and simulate xhat only over t_eval
    y_observed, t_y, xtraj_estim = \
        simulate_estimations(system=system, observe_data=observe_data,
                             t_eval=t_eval, t0=t0, tf=tf, dt=dt,
                             meas_noise_var=meas_noise_var,
                             init_control=init_control,
                             init_state_estim=init_state_estim,
                             controller=controller, observer=observer,
                             optim_method=optim_method,
                             dyn_kwargs=dyn_kwargs, xtraj=xtraj,
                             GP=observer_prior_mean, discrete=discrete,
                             verbose=verbose)

    # Create initial data for GP, noiseless or noisy X, noiseless U, noisy Y
    X, U, Y = form_GP_data(system=system, xtraj=xtraj,
                           xtraj_estim=xtraj_estim, utraj=utraj,
                           meas_noise_var=meas_noise_var, **dyn_kwargs)

    os.makedirs('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                exist_ok=False)
    for i in range(np.min([xtraj.shape[1], xtraj_estim.shape[1]])):
        name = 'xtraj_xtrajestim_' + str(i) + '.pdf'
        plt.plot(xtraj[:, i], label='True state', c='g')
        plt.plot(xtraj_estim[:, i], label='Estimated state', c='orange')
        plt.title('True and estimated position over time')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('x_' + str(i))
        plt.savefig(
            os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                         name),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
    name = 'Estimation_error'
    dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
    error = np.sum(
        np.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]), axis=1)
    df = pd.DataFrame(error)
    df.to_csv(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     name + '.csv'), header=False)
    plt.plot(error, 'orange', label='True trajectory')
    plt.title('Error plot')
    plt.xlabel('t')
    plt.ylabel(r'$|x - \hat{x}|$')
    plt.legend()
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     name + '.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()

    # Create LS model
    dyn_kwargs.update({'dt': dt, 't0': t0, 'tf': tf, 't_span': t_span,
                       'init_state': init_state,
                       'init_state_estim': init_state_estim,
                       'init_control': init_control, 'observer_prior_mean':
                           observer_prior_mean})


    def sigma(x):
        # Features used for least squares (big prior knowledge!)
        x = reshape_pt1(x)
        return np.concatenate((reshape_dim1(-x[:, 0]),
                               reshape_dim1(-x[:, 0] ** 3),
                               reshape_dim1(-x[:, 1])), axis=1)


    def dsigma_dx(x):
        # Derivative of feature vector
        x = reshape_pt1(x)
        assert x.shape[0] == 1, 'Can only deal with one point at a time'
        return np.array([[-1, 0], [-3 * x[:, 0] ** 2, 0], [0, -1]])


    # Fit and evaluate LS model: find best coeffs given training data,
    # then with same training data still see what LS model predicts
    reg = linear_model.LinearRegression()
    reg.fit(X=sigma(X), y=Y)
    if 'LS_justvelocity' in system:
        dt = dyn_kwargs.get('dt')
        true_coefs = [dyn_kwargs.get('alpha') * dt, dyn_kwargs.get('beta') * dt,
                      dyn_kwargs.get('delta') * dt - 1]
    elif 'LS_Michelangelo' in system:
        true_coefs = [dyn_kwargs.get('alpha'), dyn_kwargs.get('beta'),
                      dyn_kwargs.get('delta')]
    else:
        raise Exception('True coefs not defined for this system')
    coef_error = np.linalg.norm(reg.coef_ - true_coefs)
    specs_file = os.path.join(
        '../Figures/' + system + '/Least_squares_sigma/Loop_0',
        'Specifications.txt')
    with open(specs_file, 'w') as f:
        print('LS reg score: ', reg.score(X=sigma(X), y=Y), file=f)
        print('Data shape: ', X.shape, Y.shape, file=f)
        print('LS reg coef: ', reg.coef_, file=f)
        print('True coefs for these features: ', true_coefs, file=f)
        print('Current coef error: ', coef_error, file=f)
    plt.plot(X[:, 0], reg.predict(sigma(X)), label='xi predicted by LS')
    plt.plot(X[:, 0], Y, label='xi given by observer')
    plt.legend()
    plt.title('Predicted xi depending on x')
    if verbose:
        plt.show()
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     'X0_predict.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.plot(sigma(X)[:, 0], reg.predict(sigma(X)), label='xi predicted by LS')
    plt.plot(sigma(X)[:, 0], Y, label='xi given by observer')
    plt.legend()
    plt.title('Predicted xi depending on sigma(x)')
    if verbose:
        plt.show()
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     'Sigma0_predict.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.plot(sigma(X)[:, 1], reg.predict(sigma(X)), label='xi predicted by LS')
    plt.plot(sigma(X)[:, 1], Y, label='xi given by observer')
    plt.legend()
    plt.title('Predicted xi depending on sigma(\dot{x})')
    if verbose:
        plt.show()
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     'Sigma1_predict.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()
    coef_errors = [coef_error]
    plt.plot(coef_errors, label='Error on coef')
    plt.legend()
    plt.title('Error on coefficients of LS with current features')
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     'Parameter_errors.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()
    state_estimation_RMSE = [RMS(xtraj - xtraj_estim[:, :xtraj.shape[1]])]
    plt.plot(state_estimation_RMSE, label='RMSE')
    plt.legend()
    plt.title('State estimation RMSE')
    plt.savefig(
        os.path.join('../Figures/' + system + '/Least_squares_sigma/Loop_0',
                     'State_estimation_RMSE.pdf'),
        bbox_inches='tight')
    plt.close('all')
    plt.clf()
    current_model = lambda x, u, kwargs: reg.predict(sigma(x))
    current_model_deriv = lambda x, u, kwargs: reshape_pt1(np.dot(reg.coef_,
                                                                  dsigma_dx(x)))
    print(current_model(X[0], U[0], dyn_kwargs.get('prior_kwargs')))
    print(current_model_deriv(X[0], U[0], dyn_kwargs.get('prior_kwargs')), X[0])

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

        # Create new data, by simulating again starting from the newt
        # init_state and init_state_estim, and re-learn GP
        xtraj, utraj, t_utraj = simulate_dynamics(t_span=t_span, t_eval=t_eval,
                                                  t0=t0, dt=dt,
                                                  init_control=init_control,
                                                  init_state=init_state,
                                                  dynamics=dynamics,
                                                  controller=controller,
                                                  process_noise_var=process_noise_var,
                                                  optim_method=optim_method,
                                                  dyn_kwargs=dyn_kwargs,
                                                  discrete=discrete,
                                                  verbose=verbose)

        # Observe data: only position, observer reconstitutes velocity
        # Get observations over t_eval and simulate xhat only over t_eval
        if 'LS_Michelangelo' in system:
            LS_for_observer = current_model_deriv
        elif 'LS_justvelocity_highgain' in system:
            # LS_for_observer = current_model
            current_model_deriv = lambda x, u, kwargs: reshape_pt1(
                np.dot((reg.coef_ + [0, 0, 1]) / dt, dsigma_dx(x)))
            print(reg.coef_, reg.coef_ / dt, (reg.coef_ + [0, 0, 1]) / dt)
            LS_for_observer = current_model_deriv
        else:
            raise Exception('No LS model or not defined')
        y_observed, t_y, xtraj_estim = \
            simulate_estimations(system=system, observe_data=observe_data,
                                 t_eval=t_eval, t0=t0, tf=tf, dt=dt,
                                 meas_noise_var=meas_noise_var,
                                 init_control=init_control,
                                 init_state_estim=init_state_estim,
                                 controller=controller, observer=observer,
                                 optim_method=optim_method,
                                 dyn_kwargs=dyn_kwargs, xtraj=xtraj,
                                 GP=LS_for_observer, discrete=discrete,
                                 verbose=verbose)

        # Create initial data for GP, noiseless or noisy X, noiseless U, noisy Y
        X_old, U_old, Y_old = X, U, Y
        X, U, Y = form_GP_data(system=system, xtraj=xtraj,
                               xtraj_estim=xtraj_estim, utraj=utraj,
                               meas_noise_var=meas_noise_var, **dyn_kwargs)
        os.makedirs(
            '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
            exist_ok=False)
        for j in range(np.min([xtraj.shape[1], xtraj_estim.shape[1]])):
            name = 'xtraj_xtrajestim_' + str(j) + '.pdf'
            plt.plot(xtraj[:, j], label='True state', c='g')
            plt.plot(xtraj_estim[:, j], label='Estimated state', c='orange')
            plt.title('True and estimated position over time')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('x_' + str(j))
            plt.savefig(
                os.path.join(
                    '../Figures/' + system + '/Least_squares_sigma/Loop_' +
                    str(i),
                    name),
                bbox_inches='tight')
            plt.close('all')
            plt.clf()
        name = 'Estimation_error'
        dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
        error = np.sum(
            np.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]), axis=1)
        df = pd.DataFrame(error)
        df.to_csv(
            os.path.join(
                '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
                name + '.csv'),
            header=False)
        plt.plot(error, 'orange', label='True trajectory')
        plt.title('Error plot')
        plt.xlabel('t')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(
            os.path.join(
                '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
                name + '.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()

        X = np.concatenate((X_old, X), axis=0)
        U = np.concatenate((U_old, U), axis=0)
        Y = np.concatenate((Y_old, Y), axis=0)
        reg = linear_model.LinearRegression()
        reg.fit(X=sigma(X), y=Y)
        if 'LS_justvelocity' in system:
            dt = dyn_kwargs.get('dt')
            true_coefs = [dyn_kwargs.get('alpha') * dt,
                          dyn_kwargs.get('beta') * dt,
                          dyn_kwargs.get('delta') * dt - 1]
        elif 'LS_Michelangelo' in system:
            true_coefs = [dyn_kwargs.get('alpha'), dyn_kwargs.get('beta'),
                          dyn_kwargs.get('delta')]
        else:
            raise Exception('True coefs not defined for this system')
        coef_error = np.linalg.norm(reg.coef_ - true_coefs)
        specs_file = os.path.join(
            '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
            'Specifications.txt')
        with open(specs_file, 'w') as f:
            print('LS reg score: ', reg.score(X=sigma(X), y=Y), file=f)
            print('Data shape: ', X.shape, Y.shape, file=f)
            print('LS reg coef: ', reg.coef_, file=f)
            print('True coefs for these features: ', true_coefs, file=f)
            print('Current coef error: ', coef_error, file=f)
        coef_errors += [coef_error]
        plt.plot(coef_errors)
        plt.savefig(
            os.path.join(
                '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
                'Parameter_errors.pdf'), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        state_estimation_RMSE += [RMS(xtraj - xtraj_estim[:, :xtraj.shape[1]])]
        plt.plot(state_estimation_RMSE, label='RMSE')
        plt.legend()
        plt.title('State estimation RMSE')
        plt.savefig(
            os.path.join(
                '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
                'State_estimation_RMSE.pdf'), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.plot(X[:, 0], reg.predict(sigma(X)), label='xi predicted by LS')
        plt.plot(X[:, 0], Y, label='xi given by observer')
        plt.legend()
        plt.title('Predicted xi depending on x')
        if verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
            'X0_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.plot(sigma(X)[:, 0], reg.predict(sigma(X)),
                 label='xi predicted by LS')
        plt.plot(sigma(X)[:, 0], Y, label='xi given by observer')
        plt.legend()
        plt.title('Predicted xi depending on sigma(x)')
        if verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
            'Sigma0_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.plot(sigma(X)[:, 1], reg.predict(sigma(X)),
                 label='xi predicted by LS')
        plt.plot(sigma(X)[:, 1], Y, label='xi given by observer')
        plt.legend()
        plt.title('Predicted xi depending on sigma(\dot{x})')
        if verbose:
            plt.show()
        plt.savefig(os.path.join(
            '../Figures/' + system + '/Least_squares_sigma/Loop_' + str(i),
            'Sigma1_predict.pdf'),
            bbox_inches='tight')
        plt.close('all')
        plt.clf()
        current_model = lambda x, u, kwargs: reg.predict(sigma(x))
        current_model_deriv = lambda x, u, kwargs: reshape_pt1(np.dot(reg.coef_,
                                                                      dsigma_dx(
                                                                          x)))
        print(current_model(X[0], U[0], dyn_kwargs.get('prior_kwargs')))
        print(current_model_deriv(X[0], U[0], dyn_kwargs.get('prior_kwargs')),
              X[0])

    stop_log()
