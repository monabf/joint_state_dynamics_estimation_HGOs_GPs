import logging
import os
import shutil
import sys

import GPy
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

sys.path.append('.')

from controllers import sin_controller_1D
from dynamics import dynamics_traj, mass_spring_mass_dynamics_z, \
    mass_spring_mass_dynamics_x, mass_spring_mass_xtoz, \
    mass_spring_mass_ztox, mass_spring_mass_v, mass_spring_mass_vdot
from gain_adaptation_laws import simple_score_adapt_highgain, \
    Praly_highgain_adaptation_law
from observers import dim1_observe_data, \
    MSM_justvelocity_observer_adaptive_highgain_GP, \
    MSM_justvelocity_observer_highgain_GP, MSM_observer_Michelangelo_GP
from prior_means import MSM_continuous_to_discrete_justvelocity_prior_mean
from simulation_functions import simulate_dynamics, simulate_estimations, \
    form_GP_data
from plotting_functions import save_outside_data, plot_outside_data
from utils import reshape_pt1, reshape_dim1, interpolate, \
    reshape_dim1_tonormal, reshape_pt1_tonormal
from quasilinear_observer_GP import start_log, stop_log
from config import Config
from simple_GP_dyn import Simple_GP_Dyn

sb.set_style('whitegrid')

# Script to test quasi-linear system with observer, adding GP to learn
# nonlinear part, on a mass-spring-mass system put into the observable
# canonical form with flatness (flat output = x1)

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


def update_params_on_loop(system, config):
    if 'Duffing' in system:
        # omega = np.random.uniform(0, 2 * np.pi, 1)
        # dyn_config['omega'] = omega
        gamma = np.random.uniform(0.2, 0.9, 1)
        config.gamma = gamma
    elif 'Pendulum' in system:
        omega = np.random.uniform(1, np.pi, 1)
        config.omega = omega
        gamma = np.random.uniform(1, 5, 1)
        config.gamma = gamma
    else:
        logging.warning('No parameter update defined for this system')
    return config


if __name__ == '__main__':
    start_log()

    # General params
    config = Config(system='Continuous/Mass_spring_mass/Discrete_model/'
                           'GP_justvelocity_highgain_observer_noisy_inputs',
                    nb_samples=250,
                    t0_span=0,
                    tf_span=15,
                    t0=0,
                    tf=15,
                    hyperparam_optim='fixed_hyperparameters',
                    true_meas_noise_var=1e-8,
                    process_noise_var=1e-8,
                    optim_method='RK23',  # good solver needed for z
                    nb_loops=1,
                    nb_rollouts=10,
                    rollout_length=200,
                    rollout_controller={'random': 3, 'sin_controller_1D': 4,
                                        'null_controller': 3},
                    max_rollout_value=5,
                    sliding_window_size=3000,
                    verbose=False,
                    monitor_experiment=True,
                    multioutput_GP=False,
                    sparse=None,
                    memory_saving=False,
                    restart_on_loop=False,
                    GP_optim_method='lbfgsb',
                    meas_noise_var=1,  # 1 TODO test Steve
                    batch_adaptive_gain=None,
                    observer_prior_mean=None,
                    prior_mean=None,
                    prior_mean_deriv=None)
    t_span = [config.t0_span, config.tf_span]
    t_eval = np.linspace(config.t0, config.tf, config.nb_samples)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))

    # System params
    if 'Continuous/Mass_spring_mass' in config.system:
        config.update(dict(
            discrete=False,
            m1=1,
            m2=1,
            k1=0.3,
            k2=0.1,  # TODO test Steve
            gamma=0.4,
            omega=1.2, dt=config.dt,
            dt_before_subsampling=0.001,
            dynamics=mass_spring_mass_dynamics_z,
            dynamics_x=mass_spring_mass_dynamics_x,
            controller=sin_controller_1D,
            init_control=reshape_pt1([0])))
        # init_state_x = reshape_pt1(np.array([[0, -0.01, 0.1, 0.01]]))
        # Random initial state
        xmin = np.array([0., -0.005, 0.1, -0.005])
        xmax = np.array([0.1, 0.005, 0.2, 0.005])
        init_state_x = reshape_pt1(np.random.uniform(low=xmin, high=xmax))
        init_state = mass_spring_mass_xtoz(init_state_x, config)
        init_state_estim = np.concatenate((reshape_pt1(init_state[0, 0]),
            reshape_pt1(np.array([[0] * (init_state.shape[1]-1)]))), axis=1)
        config.update(dict(init_state_x=init_state_x,
                           init_state=init_state,
                           init_state_estim=init_state_estim))
        # Unstable limit cycle: needs g=50 or so with fhat=f to compensate
        #     init_state=reshape_pt1(np.array([[
        #         0.06328351129370620, -0.4068801516907030, 0.4852631041981480,
        #         0.3863899972738590]])),
        #     init_state_estim=reshape_pt1(np.array([[0, 0, 0, 0]])),
        #     init_control=reshape_pt1([0])))
        # config.update(dict(
        #     init_state_x=mass_spring_mass_ztox(config.init_state, config)))
        if 'GP_Michelangelo' in config.system:
            config.update(dict(
                observer=MSM_observer_Michelangelo_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10,
                                                 'k1': 5,
                                                 'k2': 5,
                                                 'k3': 5,
                                                 'k4': 2,
                                                 'k5': 1}},
                saturation=np.array([-1, 1]),
                observer_prior_mean=None,
                prior_mean=None,
                prior_mean_deriv=None,
                # init_state_estim=reshape_pt1(np.array([[0, 0, 0, 0, 0]]))))
                init_state_estim=np.concatenate((
                    config.init_state_estim, np.array([[0]])), axis=1)))
        elif 'GP_justvelocity_highgain' in config.system:
            config.update(dict(
                observer=MSM_justvelocity_observer_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains': {'g': 10,
                                                 'k1': 5,
                                                 'k2': 5,
                                                 'k3': 3,
                                                 'k4': 1}}))
            if config.continuous_model:
                config.update(dict(saturation=np.array([-1, 1]),
                                   observer_prior_mean=None,
                                   prior_mean=None,
                                   prior_mean_deriv=None))
            else:
                config.update(dict(
                    saturation=np.array([-1, 1]),
                    observer_prior_mean=None,
                    prior_mean=MSM_continuous_to_discrete_justvelocity_prior_mean,
                    prior_mean_deriv=None))
        elif 'GP_justvelocity_adaptive_highgain' in config.system:
            config.update(dict(
                observer=
                MSM_justvelocity_observer_adaptive_highgain_GP,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': 0.001,
                              'observer_gains':
                                  {'g': 10,
                                   'k1': 5,
                                   'k2': 5,
                                   'k3': 3,
                                   'k4': 1,
                                   'p1': 1e6,
                                   'p2': 5e-9,
                                   'b': 1e-4,
                                   'n': config.init_state.shape[1],
                                   'adaptation_law':
                                       Praly_highgain_adaptation_law}},
                saturation=np.array([-1, 1]),
                observer_prior_mean=None,
                prior_mean=MSM_continuous_to_discrete_justvelocity_prior_mean,
                prior_mean_deriv=None))
            config.update(dict(init_state_estim=np.concatenate((
                    config.init_state_estim,
                    reshape_pt1([config.prior_kwargs['observer_gains'].get(
                        'g')])), axis=1)))
        elif 'No_observer' in config.system:
            config.update(dict(observer=None,
                               observer_prior_mean=None))
        # Create kernel
        if config.get('gamma') == 0:
            input_dim = config.init_state.shape[1]
            no_control = True
        else:
            input_dim = config.init_state.shape[1] + \
                        config.init_control.shape[1]
            no_control = False
        kernel = GPy.kern.RBF(input_dim=input_dim, variance=3.5,
                              lengthscale=np.array([150, 150, 1.5, 2.5, 2.5]),
                              ARD=True)  # TODO test Steve
        kernel.unconstrain()
        kernel.variance.set_prior(GPy.priors.Gaussian(3.5, 3.5))
        kernel.lengthscale.set_prior(
            GPy.priors.MultivariateGaussian(np.array([150, 150, 1.5, 2.5, 2.5]),
                                            np.diag([150, 150, 1.5, 2.5, 2.5])))
        config.update(dict(observe_data=dim1_observe_data,
                           constrain_u=[-config.get('gamma'),
                                        config.get('gamma')],
                           constrain_x=[],
                           grid_inf=[-0.5, -0.5, -0.5, -0.5],
                           grid_sup=[0.5, 0.5, 0.5, 0.5],
                           input_dim=input_dim,
                           no_control=no_control,
                           kernel=kernel))
    else:
        raise Exception('Unknown system')

    # Set derivative_function for continuous models
    if config.continuous_model:
        if config.prior_mean:
            logging.warning(
                'A prior mean has been defined for the GP though '
                'a continuous model is being used. Check this is '
                'really what you want to do, as a prior mean is '
                'often known for discrete models without being '
                'available for continuous ones.')


        def derivative_function(X, U, y_observed, GP):
            X = reshape_pt1(X)
            u = lambda t, kwargs, t0, init_control: reshape_pt1(U)[t]
            y = lambda t, kwargs: reshape_pt1(y_observed)[t]
            # time = np.arange(len(X))
            # obs_1D = lambda t: config.observer(t, X[t], u, y, config.t0,
            #                                    config.init_control, GP, config)
            # Xdot = np.apply_along_axis(func1d=obs_1D, axis=0, arr=time)
            Xdot = np.array([config.observer(t, X[t], u, y, config.t0,
                                             config.init_control, GP,
                                             config) for t in
                             range(len(X))])
            Xdot = Xdot.reshape(X.shape)
            return Xdot.reshape(X.shape)


        config.update(dict(derivative_function=derivative_function))
    else:
        config.update(dict(derivative_function=None))
    print(config)

    # Check valid config
    if (config.m1 < 0) or (config.m2 < 0) or (config.k1 < 0) or (config.k2 < 0):
        raise ValueError('All physical parameters should be positive for this '
                         'solution of the MSM system to be valid: '
                         'http://eqworld.ipmnet.ru/en/solutions/ae/ae0103.pdf')

    # Loop to create data and learn GP several times
    for loop in range(config.nb_loops):

        # Adapt parameters on loop start if necessary
        if loop > 0:
            # Update params and initial states after the first pass
            if config.restart_on_loop:
                config = update_params_on_loop(config.system, config)
                t0 = config.t0
                tf = config.tf
                t0_span = config.t0_span
                tf_span = config.tf_span
            else:
                init_state_x = reshape_pt1(xtraj_orig_coord[-1])
                init_state = reshape_pt1(xtraj[-1])
                init_control = reshape_pt1(utraj[-1])
                init_state_estim = reshape_pt1(xtraj_estim[-1])
                tf_before = config.tf
                tf_span = tf_before + (config.tf_span - config.t0_span)
                t0_span = tf_before
                tf = tf_before + (config.tf - config.t0)
                t0 = tf_before
                config.update(dict(t0=t0, tf=tf, t0_span=t0_span,
                                   tf_span=tf_span, init_state_x=init_state_x,
                                   init_state=init_state,
                                   init_control=init_control,
                                   init_state_estim=init_state_estim))
            t_span = [t0_span, tf_span]
            t_eval = np.linspace(t0, tf, config.nb_samples)

            # Update observer gain
            if config.batch_adaptive_gain:
                if 'simple_score_posdist_lastbatch' in config.batch_adaptive_gain:
                    gain = config.prior_kwargs['observer_gains'].get('g')
                    score = np.linalg.norm(
                        reshape_dim1_tonormal(xtraj_estim[:, 0] - xtraj[:, 0]))
                    previous_idx = int(np.min([loop, 2]))
                    (base_path, previous_loop) = os.path.split(
                        dyn_GP.results_folder)
                    previous_results_folder = os.path.join(
                        base_path, 'Loop_' + str(loop - previous_idx))
                    previous_xtraj_estim = pd.read_csv(os.path.join(
                        previous_results_folder,
                        'Data_outside_GP/xtraj_estim.csv'),
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
                    config.prior_kwargs['observer_gains']['g'] = new_gain
                    gain_time = np.concatenate(
                        (gain_time, np.array([new_gain])))
                elif 'simple_score_' in config.batch_adaptive_gain:
                    param = \
                        config.batch_adaptive_gain.split('simple_score_', 1)[1]
                    gain = config.prior_kwargs['observer_gains'].get('g')
                    score = dyn_GP.variables[param][-1, 1]
                    previous_idx = int(np.min([loop, 2]))
                    previous_score = dyn_GP.variables[param][-previous_idx, 1]
                    new_gain = simple_score_adapt_highgain(gain, score,
                                                           previous_score)
                    config.prior_kwargs['observer_gains']['g'] = new_gain
                    gain_time = np.concatenate(
                        (gain_time, np.array([new_gain])))
                elif config.batch_adaptive_gain == 'change_last_batch':
                    if loop == config.nb_loops - 1:
                        new_gain = 3
                        config.prior_kwargs['observer_gains']['g'] = new_gain
                        gain_time = np.concatenate(
                            (gain_time, np.array([new_gain])))
                else:
                    logging.error(
                        'This adaptation law for the observer gains has '
                        'not been defined.')

        # Simulate system in x
        xtraj_orig_coord, utraj, t_utraj = \
            simulate_dynamics(t_span=t_span, t_eval=t_eval,
                              t0=config.t0, dt=config.dt,
                              init_control=config.init_control,
                              init_state=config.init_state_x,
                              dynamics=config.dynamics_x,
                              controller=config.controller,
                              process_noise_var=config.process_noise_var,
                              optim_method=config.optim_method,
                              dyn_config=config,
                              discrete=config.discrete,
                              verbose=config.verbose)
        # Plot trajectories
        plt.plot(xtraj_orig_coord[:, 0], label='x1')
        plt.plot(xtraj_orig_coord[:, 2], label='x2')
        plt.title('Positions over time')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(xtraj_orig_coord[:, 1], label=r'$\dot{x_1}$')
        plt.plot(xtraj_orig_coord[:, 3], label=r'$\dot{x_2}$')
        plt.title('Velocities over time')
        plt.xlabel('t')
        plt.ylabel(r'$\dot{x}$')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(xtraj_orig_coord[:, 0], xtraj_orig_coord[:, 1], label='x1')
        plt.plot(xtraj_orig_coord[:, 2], xtraj_orig_coord[:, 3], label='x2')
        plt.title('Phase portraits')
        plt.xlabel('x')
        plt.ylabel(r'$\dot{x}$')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(utraj, label='u')
        plt.title('Control trajectory')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.legend()
        if config.verbose:
            plt.show()

        # Simulate corresponding system in z
        xtraj, utraj, t_utraj = \
            simulate_dynamics(t_span=t_span, t_eval=t_eval,
                              t0=config.t0, dt=config.dt,
                              init_control=config.init_control,
                              init_state=config.init_state,
                              dynamics=config.dynamics,
                              controller=config.controller,
                              process_noise_var=config.process_noise_var,
                              optim_method=config.optim_method,
                              dyn_config=config,
                              discrete=config.discrete,
                              verbose=config.verbose)
        # Plot trajectories
        for i in range(xtraj.shape[1]):
            plt.plot(xtraj[:, i], label='z_' + str(i))
            plt.title('z derived ' + str(i) + ' times')
            plt.xlabel('t')
            plt.ylabel('z_' + str(i))
            plt.legend()
            if config.verbose:
                plt.show()
        plt.plot(utraj, label='u')
        plt.title('Control trajectory')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.legend()
        if config.verbose:
            plt.show()

        # Check invertibility of transformation
        plt.plot(xtraj_orig_coord[:, 0], label='x')
        plt.plot(mass_spring_mass_ztox(xtraj, config)[:, 0], label='T*('
                                                                   'z)')
        plt.plot(
            mass_spring_mass_ztox(
                mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 0], label='T*(T(x))')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(xtraj_orig_coord[:, 1], label='x')
        plt.plot(mass_spring_mass_ztox(xtraj, config)[:, 1],
                 label='T*(z)')
        plt.plot(
            mass_spring_mass_ztox(
                mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 1], label='T*(T(x))')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(xtraj_orig_coord[:, 2], label='x')
        plt.plot(mass_spring_mass_ztox(xtraj, config)[:, 2],
                 label='T*(z)')
        plt.plot(
            mass_spring_mass_ztox(
                mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 2], label='T*(T(x))')
        plt.legend()
        if config.verbose:
            plt.show()
        plt.plot(xtraj_orig_coord[:, 3], label='x')
        plt.plot(mass_spring_mass_ztox(xtraj, config)[:, 3],
                 label='T*(z)')
        plt.plot(
            mass_spring_mass_ztox(
                mass_spring_mass_xtoz(xtraj_orig_coord, config),
                config)[:, 3], label='T*(T(x))')
        plt.legend()
        if config.verbose:
            plt.show()

        # Define true dynamics
        if ('justvelocity' in config.system) and (
                'Mass_spring_mass' in config.system):
            if config.continuous_model:
                true_dynamics = lambda x, control: \
                    config.dynamics(t=config.t0, z=x, u=lambda t, kwarg, t0,
                                                               init_control:
                    interpolate(t, np.concatenate((reshape_dim1(np.arange(
                        len(control))), control), axis=1), t0=t0,
                                init_value=init_control),
                                    t0=config.t0, init_control=control,
                                    process_noise_var=config.process_noise_var,
                                    kwargs=config)[:, -1]
            else:
                true_dynamics = lambda x, control: dynamics_traj(
                    x0=reshape_pt1(x), u=lambda t, kwarg, t0, init_control:
                    interpolate(t, np.concatenate((reshape_dim1(np.arange(
                        len(control))), control), axis=1),
                                t0=t0, init_value=init_control),
                    t0=config.t0, dt=config.dt,
                    init_control=config.init_control,
                    version=config.dynamics,
                    meas_noise_var=0,
                    process_noise_var=config.process_noise_var,
                    method=config.optim_method, t_span=[0, config.dt],
                    t_eval=[config.dt],
                    kwargs=config)[:, -1]
        elif ('Michelangelo' in config.system) and (
                'Mass_spring_mass' in config.system):
            def true_dynamics(x, control):
                m1 = config.get('m1')
                m2 = config.get('m2')
                k1 = config.get('k1')
                k2 = config.get('k2')
                z = reshape_pt1(x)
                z3 = reshape_pt1(z[:, 2])
                u = control
                v = reshape_pt1_tonormal(mass_spring_mass_v(z, config))
                vdot = reshape_pt1_tonormal(
                    mass_spring_mass_vdot(z, config))
                return reshape_pt1(
                    k1 * (m1 * m2) * (u - (m1 + m2) * z3) + (3 * k2) / (
                            m1 * m2) * (u - (
                            m1 + m2) * z3) * v ** 2 + (
                            6 * k2) / m1 * v * vdot ** 2)
        elif (('Michelangelo' in config.system) or (
                'justvelocity_highgain' in config.system)) \
                and not any(
            k in config.system for k in ('Duffing', 'Harmonic_oscillator',
                                         'Pendulum', 'VanderPol')):
            raise Exception('No ground truth has been defined.')
        config.update(dict(true_dynamics=true_dynamics))

        # Observe data: only position, observer reconstitutes velocity
        # Get observations over t_eval and simulate xhat only over t_eval
        if loop == 0:
            observer_prior_mean = config.observer_prior_mean
        else:
            observer_prior_mean = dyn_GP
        y_observed, t_y, xtraj_estim = \
            simulate_estimations(system=config.system,
                                 observe_data=config.observe_data,
                                 t_eval=t_eval, t0=config.t0, tf=config.tf,
                                 dt=config.dt,
                                 meas_noise_var=config.true_meas_noise_var,
                                 init_control=config.init_control,
                                 init_state_estim=config.init_state_estim,
                                 controller=config.controller,
                                 observer=config.observer,
                                 optim_method=config.optim_method,
                                 dyn_config=config, xtraj=xtraj,
                                 GP=observer_prior_mean,
                                 discrete=config.discrete,
                                 verbose=config.verbose)

        # Create data for GP, noiseless or noisy X, noiseless U, noisy Y
        X, U, Y = form_GP_data(system=config.system, xtraj=xtraj,
                               xtraj_estim=xtraj_estim, utraj=utraj,
                               meas_noise_var=config.true_meas_noise_var,
                               y_observed=y_observed,
                               derivative_function=config.derivative_function,
                               model=observer_prior_mean)

        # Initialize or re-initialize GP
        if loop == 0:
            dyn_GP = Simple_GP_Dyn(X, U, Y, config)
        else:
            (base_path, previous_loop) = os.path.split(dyn_GP.results_folder)
            new_results_folder = os.path.join(base_path, 'Loop_' + str(loop))
            os.makedirs(new_results_folder, exist_ok=False)
            dyn_GP.set_results_folder(new_results_folder)
            dyn_GP.set_config(config)

        # Set up outside data to save
        data_to_save = {'xtraj_orig_coord': xtraj_orig_coord, 'xtraj': xtraj,
                        'xtraj_estim': xtraj_estim,
                        'transfo_ztox_xtoz_xtraj_orig_coord':
                            mass_spring_mass_ztox(mass_spring_mass_xtoz(
                                xtraj_orig_coord, config), config),
                        'transfo_ztox_xtraj': mass_spring_mass_ztox(xtraj,
                                                                    config),
                        'y_observed': y_observed}
        if config.batch_adaptive_gain:
            gain_time = np.array(
                config.prior_kwargs['observer_gains']['g'])
            data_to_save.update({'gain_time': gain_time})
        elif 'adaptive' in config.system:
            output_error = reshape_dim1(
                np.square(xtraj[:, 0] - xtraj_estim[:, 0]))
            if loop == 0:
                gain_time = reshape_dim1(xtraj_estim[:, -1])
            else:
                gain_time = np.concatenate((
                    gain_time, reshape_dim1(xtraj_estim[:, -1])))
            data_to_save.update(
                {'gain_time': gain_time, 'output_error': output_error})

        # Train GP with estimated trajectory, evaluate and save results
        save_outside_data(dyn_GP, data_to_save)
        plot_outside_data(dyn_GP, data_to_save)
        if loop == 0:
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
                    MSM_justvelocity_observer_highgain_GP,
                    config.observe_data, no_GP_in_observer=True)
                if config.prior_mean:
                    dyn_GP.evaluate_kalman_rollouts(
                        MSM_justvelocity_observer_highgain_GP,
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

        else:
            dyn_GP.learn(new_X=X, new_Y=Y, new_U=U)
        dyn_GP.save()
        if 'justvelocity_adaptive' in config.system:
            # Do not adapt observer gains for closed-loop rollouts
            dyn_GP.evaluate_kalman_rollouts(
                MSM_justvelocity_observer_highgain_GP, config.observe_data,
                config.discrete)
            dyn_GP.evaluate_closedloop_rollouts(
                MSM_justvelocity_observer_highgain_GP, config.observe_data)
        else:
            dyn_GP.evaluate_kalman_rollouts(config.observer,
                                            config.observe_data,
                                            config.discrete)
            dyn_GP.evaluate_closedloop_rollouts(config.observer,
                                                config.observe_data)

        stop_log()
