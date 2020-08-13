import numpy as np

from dynamics import mass_spring_mass_v, mass_spring_mass_vdot, dynamics_traj
from utils import reshape_pt1, reshape_dim1, reshape_pt1_tonormal


# Prior mean functions for several systems, of form f0(x, u, prior_kwargs)

# Prior mean for continuous time Duffing equation: same dynamics as oscillator
# but with prior params and fixed x,u. Gives xdot then can be integrated
def duffing_continuous_prior_mean(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    # Amult = np.array([np.dot(A, x[i, :]) for i in range(len(x))])
    dot = lambda a: np.dot(A, a)
    Amult = np.apply_along_axis(func1d=dot, axis=1, arr=x)
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(- beta * (x[:, 0]) ** 3)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xdot = reshape_pt1(Amult + F + u)
    return xdot


# Prior mean for discrete time Duffing equation: same dynamics as oscillator
# but with prior params and fixed x, u. Gives xnext for discrete GP model
def duffing_continuous_to_discrete_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                duffing_continuous_prior_mean(xl, ul, kwargs)
        xnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)
        return reshape_pt1_tonormal(xnext)

    xnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # xnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   duffing_continuous_prior_mean(xl, ul, kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])
    return xnext


# Prior mean for discrete time Duffing map, gives xnext directly
def duffing_discrete_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                duffing_continuous_prior_mean(xl, ul, kwargs)
        xnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)
        return reshape_pt1_tonormal(xnext)

    xnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # xnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   duffing_continuous_prior_mean(xl, ul, kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])
    return xnext


# Prior mean of model deriv for continuous time Duffing equation extended
# for Michelangelo's extended high gain observer framework.  Gives d
# phihat_prior / dx
def duffing_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    deriv = reshape_dim1(- 3 * beta * ((x[:, 0]) ** 2) - alpha)
    phi_deriv = reshape_pt1(np.concatenate((
        deriv, reshape_dim1(- delta * np.ones_like(x[:, 0]))), axis=1))
    return phi_deriv


def duffing_continuous_prior_mean_Michelangelo_deriv_u(x, u, prior_kwargs):
    deriv = duffing_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat)
def duffing_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    phi = reshape_dim1(
        - beta * ((x[:, 0]) ** 3) - alpha * x[:, 0] - delta * x[:, 1])
    return phi


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat) + Du part
def duffing_continuous_prior_mean_Michelangelo_u(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    phi = reshape_dim1(
        - beta * ((x[:, 0]) ** 3) - alpha * x[:, 0] - delta * x[:, 1] + u[:, 1])
    return phi


# Prior mean of model deriv for continuous time pendulmum equation extended
# for Michelangelo's extended high gain observer framework.  Gives d
# phihat_prior / dx
def pendulum_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    g = prior_kwargs.get('g')
    l = prior_kwargs.get('l')
    x = reshape_pt1(x)
    deriv = reshape_dim1(- g / l * np.cos(x[:, 0]))
    phi_deriv = reshape_pt1(np.concatenate((
        deriv, reshape_dim1(- k / m * np.ones_like(x[:, 0]))), axis=1))
    return phi_deriv


def pendulum_continuous_prior_mean_Michelangelo_deriv_u(x, u, prior_kwargs):
    deriv = duffing_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean for continuous time pendulum equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat)
def pendulum_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    g = prior_kwargs.get('g')
    l = prior_kwargs.get('l')
    x = reshape_pt1(x)
    phi = reshape_dim1(- g / l * np.sin(x[:, 0]) - k / m * x[:, 1])
    return phi


# Prior mean for continuous time pendulum equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat) + Du part
def pendulum_continuous_prior_mean_Michelangelo_u(x, u, prior_kwargs):
    return pendulum_continuous_prior_mean_Michelangelo(x, u, prior_kwargs) + \
           reshape_pt1(u[:, 1])


# Prior mean of model deriv for continuous time Duffing equation extended
# for Michelangelo's extended high gain observer framework.  Gives d
# phihat_prior / dx, for modified Duffing version with nonlinearity cos(x)**2
# instead of x**3
def duffing_cossquare_continuous_prior_mean_Michelangelo_deriv(x, u,
                                                               prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    deriv = reshape_dim1(2 * beta * (np.cos(x[:, 0]) * np.sin(x[:, 0])) - alpha)
    phi_deriv = reshape_pt1(np.concatenate((
        deriv, reshape_dim1(- delta * np.ones_like(x[:, 0]))), axis=1))
    return phi_deriv


def duffing_cossquare_continuous_prior_mean_Michelangelo_deriv_u(x, u,
                                                                 prior_kwargs):
    deriv = duffing_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat), for modified
# Duffing version with nonlinearity cos(x)**2 instead of x**3
def duffing_cossquare_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    phi = reshape_dim1(
        - beta * (np.cos(x[:, 0]) ** 2) - alpha * x[:, 0] - delta * x[:, 1])
    return phi


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat), for modified
# Duffing version with nonlinearity cos(x)**2 instead of x**3 + Du part
def duffing_cossquare_continuous_prior_mean_Michelangelo_u(x, u, prior_kwargs):
    alpha = prior_kwargs.get('alpha')
    beta = prior_kwargs.get('beta')
    delta = prior_kwargs.get('delta')
    x = reshape_pt1(x)
    phi = reshape_dim1(
        - beta * (np.cos(x[:, 0]) ** 2) - alpha * x[:, 0] - delta * x[:, 1] +
        u[:, 1])
    return phi


# Prior mean for continuous harmonic oscillator: same dynamics as oscillator
# but with prior params and fixed x,u. Gives xdot then can be integrated
def harmonic_oscillator_continuous_prior_mean(x, u, prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    A = reshape_pt1([[0, 1], [- k / m, 0]])
    # Amult = np.array([np.dot(A, x[i, :]) for i in range(len(x))])
    dot = lambda a: np.dot(A, a)
    Amult = np.apply_along_axis(func1d=dot, axis=1, arr=x)
    xdot = reshape_pt1(Amult + u)
    return xdot


# Prior mean for continuous harmonic oscillator: same dynamics as oscillator
# but with prior params and fixed x,u. Gives xdot then can be integrated
def harmonic_oscillator_continuous_to_discrete_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                harmonic_oscillator_continuous_prior_mean(xl, ul, kwargs)
        xnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)
        return reshape_pt1_tonormal(xnext)

    xnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # xnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   harmonic_oscillator_continuous_prior_mean(xl, ul, kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])
    return xnext


# Prior mean of model deriv for continuous time HO equation extended
# for Michelangelo's extended high gain observer framework. Gives d
# phihat_prior / dx
def harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv(x, u,
                                                                 prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    deriv = reshape_dim1(- k / m * np.ones_like(x[:, 0]))
    phi_deriv = reshape_pt1(
        np.concatenate((deriv, np.zeros_like(deriv)), axis=1))
    return phi_deriv


def harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv_u(
        x, u, prior_kwargs):
    deriv = harmonic_oscillator_continuous_prior_mean_Michelangelo_deriv(
        x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat)
def harmonic_oscillator_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    x = reshape_pt1(x)
    phi = reshape_dim1(- k / m * x[:, 0])
    return phi


# Prior mean for continuous time Duffing equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat) + Du part
def harmonic_oscillator_continuous_prior_mean_Michelangelo_u(x, u,
                                                             prior_kwargs):
    k = prior_kwargs.get('k')
    m = prior_kwargs.get('m')
    x = reshape_pt1(x)
    phi = reshape_dim1(- k / m * x[:, 0] + u[:, 1])
    return phi


# Prior mean of model deriv for continuous time VanderPol equation extended
# for Michelangelo's extended high gain observer framework.  Gives d
# phihat_prior / dx
def VanderPol_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs):
    mu = prior_kwargs.get('mu')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    deriv_x = reshape_dim1(-2 * mu * x[:, 0] * x[:, 1] - np.ones_like(x[:, 0]))
    deriv_xdot = reshape_dim1(mu * (1 - x[:, 0] ** 2))
    phi_deriv = reshape_pt1(np.concatenate((deriv_x, deriv_xdot), axis=1))
    return phi_deriv


def VanderPol_continuous_prior_mean_Michelangelo_deriv_u(
        x, u, prior_kwargs):
    deriv = VanderPol_continuous_prior_mean_Michelangelo_deriv(
        x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean for continuous time VanderPol equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat)
def VanderPol_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    mu = prior_kwargs.get('mu')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    phi = reshape_dim1(mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0])
    return phi


# Prior mean for continuous time VanderPol equation extended for Michelangelo's
# extended high gain observer framework. Gives Phihat(xhat) + Du part
def VanderPol_continuous_prior_mean_Michelangelo_u(x, u, prior_kwargs):
    mu = prior_kwargs.get('mu')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    phi = reshape_dim1(mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0] + u[:, 1])
    return phi


# Prior mean of continuous time model of Wandercraft's arm: just prior
# dynamics of arm for given (x, u). Gives xdot then can be integrated with Euler
def wdc_arm_continuous_prior_mean(x, u, prior_kwargs):
    inertia = prior_kwargs.get('inertia')
    m = prior_kwargs.get('m')
    lG = prior_kwargs.get('lG')
    g = prior_kwargs.get('g')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    A = reshape_pt1([[0, 1], [0, 0]])
    # Amult = np.array([np.dot(A, x[i, :]) for i in range(len(x))])
    dot = lambda a: np.dot(A, a)
    Amult = np.apply_along_axis(func1d=dot, axis=1, arr=x)
    # F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    # F2 = reshape_dim1(m * g * lG / inertia * np.sin(x[:, 0]))
    # F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    F = reshape_pt1(np.zeros_like(u))
    xdot = reshape_pt1(Amult + F + u)
    return xdot


# Prior mean of continuous time model of Wandercraft's arm: just prior
# dynamics of arm for given (x, u). Gives xnext for discrete GP model
def wdc_arm_continuous_to_discrete_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                wdc_arm_continuous_prior_mean(xl, ul, kwargs)
        xnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)
        return reshape_pt1_tonormal(xnext)

    xnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # xnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   wdc_arm_continuous_prior_mean(xl, ul, kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])
    return xnext


# Prior mean of continuous time model of Wandercraft's arm: just prior
# dynamics of arm for given (x, u). Gives xdot for continuous GP model
def wdc_arm_continuous_justvelocity_prior_mean(x, u, prior_kwargs):
    inertia = prior_kwargs.get('inertia')
    m = prior_kwargs.get('m')
    lG = prior_kwargs.get('lG')
    g = prior_kwargs.get('g')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    # F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    # F2 = reshape_dim1(-m * g * lG / inertia * np.sin(x[:, 0]))
    # F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    F = reshape_pt1(np.zeros_like(u))
    vdot = reshape_dim1(F[:, 1] + u[:, 1])
    return vdot


# Prior mean of continuous time model of Wandercraft's arm: just prior
# dynamics of arm for given (x, u). Gives xnext for discrete GP model
def wdc_arm_continuous_to_discrete_justvelocity_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                wdc_arm_continuous_justvelocity_prior_mean(xl, ul, kwargs)
        vnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)[:, -1]
        return reshape_pt1_tonormal(vnext)

    vnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # vnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   wdc_arm_continuous_justvelocity_prior_mean(xl, ul,
    #                                                              kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])[:, -1]
    return reshape_dim1(vnext)


# Prior mean for continuous time model of Wandercraft's arm testbed extended
# for Michelangelo's extended high gain observer framework.
# Gives Phihat(xhat)
def wdc_arm_continuous_prior_mean_Michelangelo(x, u, prior_kwargs):
    inertia = prior_kwargs.get('inertia')
    m = prior_kwargs.get('m')
    lG = prior_kwargs.get('lG')
    g = prior_kwargs.get('g')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    # phi = reshape_dim1(-m * g * lG / inertia * np.sin(x[:, 0]))
    phi = reshape_dim1(np.zeros_like(x[:, 0]))
    return phi


# Prior mean for continuous time model of Wandercraft's arm testbed extended
# for Michelangelo's extended high gain observer framework.
# Gives Phihat(xhat) + Du part
def wdc_arm_continuous_prior_mean_Michelangelo_u(x, u, prior_kwargs):
    inertia = prior_kwargs.get('inertia')
    m = prior_kwargs.get('m')
    lG = prior_kwargs.get('lG')
    g = prior_kwargs.get('g')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    # phi = reshape_dim1(-m * g * lG / inertia * np.sin(x[:, 0]))
    phi = reshape_dim1(np.zeros_like(x[:, 0]) + u[:, 1])
    return phi


# Prior mean of model deriv for continuous time model of Wandercraft's arm
# testbed, extended for Michelangelo's extended high gain observer framework.
# Gives d phihat_prior / dx
def wdc_arm_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs):
    x = reshape_pt1(x)
    phi_deriv = reshape_pt1(np.zeros_like(x))
    return phi_deriv


def wdc_arm_continuous_prior_mean_Michelangelo_deriv_u(x, u, prior_kwargs):
    deriv = wdc_arm_continuous_prior_mean_Michelangelo_deriv(x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv


# Prior mean of continuous time mass-spring-mass system: just prior
# dynamics of chain of integrators. Gives xdot for continuous GP model
def MSM_continuous_justvelocity_prior_mean(x, u, prior_kwargs):
    u = reshape_pt1(u)
    F = reshape_pt1(np.zeros_like(u))
    vdot = reshape_dim1(F[:, -1])
    return vdot


# Prior mean of continuous time mass-spring-mass system: just prior
# dynamics of chain of integrators. Gives xnext for discrete GP model
def MSM_continuous_to_discrete_justvelocity_prior_mean(x, u, prior_kwargs):
    dt = prior_kwargs.get('dt')
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    xu = np.concatenate((x, u), axis=1)

    def dyns_1D(a):
        x0 = reshape_pt1(a[:x.shape[1]])
        u0 = reshape_pt1(a[x.shape[1]:])
        version = \
            lambda t, xl, ul, t0, init_control, process_noise_var, **kwargs: \
                MSM_continuous_justvelocity_prior_mean(xl, ul, kwargs)
        vnext = dynamics_traj(x0=x0, u=u0, t0=0, dt=dt, init_control=u0,
                              discrete=False, version=version,
                              meas_noise_var=0, process_noise_var=0,
                              method='RK45', t_span=[0, dt], t_eval=[dt],
                              kwargs=prior_kwargs)[:, -1]
        return reshape_pt1_tonormal(vnext)

    vnext = np.apply_along_axis(func1d=dyns_1D, axis=1, arr=xu)
    # vnext = np.array([reshape_pt1_tonormal(
    #     dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=dt, init_control=u[i, :],
    #                   discrete=False,
    #                   version=lambda t, xl, ul, t0, init_control,
    #                                  process_noise_var, **kwargs:
    #                   MSM_continuous_justvelocity_prior_mean(xl, ul, kwargs),
    #                   meas_noise_var=0, process_noise_var=0, method='RK45',
    #                   t_span=[0, dt], t_eval=[dt], kwargs=prior_kwargs))
    #     for i in range(len(x))])[:, -1]
    return reshape_dim1(vnext)


# Prior for continuous time mass-spring-mass system: gives Phi(xhat)
# nonlinearity for Michelangelo observer
def MSM_continuous_Michelangelo_prior_mean_u(x, u, prior_kwargs):
    x = reshape_pt1(x)
    u = reshape_pt1(u)
    m1 = prior_kwargs.get('m1')
    m2 = prior_kwargs.get('m2')
    k1 = prior_kwargs.get('k1')
    k2 = prior_kwargs.get('k2')
    z = reshape_pt1(x)
    z3 = reshape_pt1(z[:, 2])
    v = reshape_pt1_tonormal(mass_spring_mass_v(z, prior_kwargs))
    vdot = reshape_pt1_tonormal(mass_spring_mass_vdot(z, prior_kwargs))
    # phi = reshape_pt1(
    #     k1 * (m1 * m2) * (u - (m1 + m2) * z3) + (3 * k2) / (m1 * m2) * (
    #             u - (m1 + m2) * z3) * v ** 2 + (
    #             6 * k2) / m1 * v * vdot ** 2)
    phi = np.zeros((x.shape[0], 1))
    return phi


# Prior deriv for continuous time mass-spring-mass system: gives d Phi(xhat)
# / dx of nonlinearity for Michelangelo observer
def MSM_continuous_Michelangelo_prior_mean_deriv(x, u, prior_kwargs):
    x = reshape_pt1(x)
    phi = reshape_pt1(np.zeros_like(x))
    return phi


def MSM_continuous_Michelangelo_prior_mean_deriv_u(x, u, prior_kwargs):
    deriv = MSM_continuous_Michelangelo_prior_mean_deriv(
        x, u, prior_kwargs)
    phi_deriv = np.concatenate(
        (deriv, np.zeros((deriv.shape[0], u.shape[1]))), axis=1)
    return phi_deriv
