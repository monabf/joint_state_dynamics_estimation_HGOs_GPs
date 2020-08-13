import logging

import numpy as np
from scipy.integrate import solve_ivp

from utils import reshape_pt1, reshape_pt1_tonormal, reshape_dim1


# Possible observers (dynamics functions f(x_estim, u)) and functions to
# produce measured data from true data

# Input x, u, version and parameters, output x at the next step (dt
# later) with scipy ODE solver
def dynamics_traj_observer(x0, u, y, t0, dt, init_control, discrete=False,
                           version=None, method='RK45', t_span=[0, 1],
                           t_eval=[0.1], GP=None, **kwargs):
    if discrete:
        xtraj = np.zeros((len(t_eval), x0.shape[1]))
        xtraj[0] = reshape_pt1(x0)
        t = t0
        i = 0
        while (i < len(t_eval) - 1) and (t < t_eval[-1]):
            i += 1
            xnext = reshape_pt1(
                version(t, xtraj[-1], u, y, t0, init_control, GP, **kwargs))
            xtraj[i] = xnext
            t += dt
    else:
        sol = solve_ivp(
            lambda t, x: version(t, x, u, y, t0, init_control, GP, **kwargs),
            t_span=t_span, y0=reshape_pt1_tonormal(x0), method=method,
            t_eval=t_eval)
        xtraj = reshape_pt1(sol.y.T)
    return reshape_pt1(xtraj)


# Observer for the continuous time Duffing equation
# Source: Observer design for the duffing equation using Gersgorin’s theorem,
# by Alberto Delgado
def duffing_observer_Delgado(t, xhat, u, y, t0, init_control, GP, kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # My gains
    l1 = delta - 5
    l2 = alpha - delta ** 2 + 3 * beta * xhat[:, 0] ** 2
    # # Delgado gains
    # l1 = - 5
    # l2 = - (alpha + 3 * xhat[:, 0] ** 2)
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    F1 = reshape_dim1(np.zeros_like(xhat[:, 0]))
    F2 = reshape_dim1(- beta * xhat[:, 0] ** 3)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(xhat)) + F + LC + u)
    return xhatdot


# Observer for the discrete time Duffing map
# Source: Observer design for the duffing equation using Gersgorin’s theorem,
# by Alberto Delgado
def duffing_observer_Delgado_discrete(t, xhat, u, y, t0, init_control, GP,
                                      kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # My gains
    l1 = -0.1
    l2 = -0.1
    # # Delgado gains
    # l1 = - 5
    # l2 = - (alpha + 3 * xhat[:, 0] ** 2)
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    F1 = reshape_dim1(np.zeros_like(xhat[:, 0]))
    F2 = reshape_dim1(- beta * xhat[:, 0] ** 3)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatnext = reshape_pt1(np.dot(A, reshape_pt1_tonormal(xhat)) + F + LC + u)
    return xhatnext


# Observer for the continuous time Duffing equation
# Using current GP estimation of dynamics for xhat_t+1, + the Delgado gains (
# artificially well chosen for now since form chosen by knowing the
# dynamics...) * (xhat_t - y_t)
def duffing_observer_Delgado_GP(t, xhat, u, y, t0, init_control, GP, kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # My gains
    l1 = delta - 5
    l2 = alpha - delta ** 2 + 3 * beta * xhat[:, 0] ** 2
    # # Delgado gains
    # l1 = - 5
    # l2 = - (alpha + 3 * xhat[:, 0] ** 2)
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # In this case we have continuous observer dynamics, but GP is
                # discrete # TODO better than Euler?
                mean = (mean - xhat) / GP.prior_kwargs.get('dt')
            GP.prior_kwargs['observer_gains'] = {'l1': l1, 'l2': l2}
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u),
                      kwargs.get('prior_kwargs'))
    else:
        mean = np.zeros_like(u)
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatdot = reshape_pt1(mean + LC)
    return xhatdot


# Observer for the discrete time Duffing map
# Using current GP estimation of dynamics for xhat_t+1, + the Delgado gains (
# artificially well chosen for now since form chosen by knowing the
# dynamics...) * (xhat_t - y_t)
def duffing_observer_Delgado_GP_discrete(t, xhat, u, y, t0, init_control, GP,
                                         kwargs):
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # My gains
    l1 = -0.8
    l2 = -0.8
    # # Delgado gains
    # l1 = - 5
    # l2 = - (alpha + 3 * xhat[:, 0] ** 2)
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u),
                      kwargs.get('prior_kwargs'))
    else:
        mean = np.zeros_like(u)
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatnext = reshape_pt1(mean + LC)
    return xhatnext


# High gain extended observer for the continuous time Duffing equation
# Using current GP estimation of dynamics for xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
def duffing_observer_Michelangelo_GP(t, xhat, u, y, t0, init_control, GP,
                                     kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    xi = reshape_pt1(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    Gamma2 = reshape_pt1([k3 * g ** 3])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean_deriv, var_deriv, lowconf_deriv, uppconf_deriv = \
                GP.predict_deriv(reshape_pt1(xhat), reshape_pt1(u), only_x=True)
            GP.prior_kwargs['observer_gains'].update({'g': g, 'Gamma1': Gamma1,
                                                      'Gamma2': Gamma2,
                                                      'k1': k1, 'k2': k2,
                                                      'k3': k3})
        else:
            mean_deriv = GP(reshape_pt1(xhat), reshape_pt1(u),
                            kwargs.get('prior_kwargs'))
    else:
        mean_deriv = np.zeros_like(xhat)
    if np.any(kwargs.get('saturation')):
        # Saturate the derivative of the nonlinearity estimate to guarantee
        # contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean_deriv = np.clip(mean_deriv, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(xi))
    DfA = reshape_pt1(np.dot(reshape_pt1_tonormal(mean_deriv),
                             reshape_pt1_tonormal(ABmult + u)))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    LC2 = reshape_pt1(Gamma2 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)
    xidot = reshape_pt1(DfA + LC2)
    # Also check eigenvalues of M for stability without high gain
    AB = np.concatenate((A, B), axis=1)
    ABO = np.concatenate((AB, np.zeros_like(reshape_pt1(AB[0]))), axis=0)
    K = np.array([[k1, k2, k3]])
    C = np.zeros_like(x)
    C[0, 0] = 1
    M = ABO - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return np.concatenate((xhatdot, xidot), axis=1)


# High gain extended observer for the continuous time Duffing equation
# Using current LS estimationfor xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
def duffing_observer_Michelangelo_LS(t, xhat, u, y, t0, init_control, LS_deriv,
                                     kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    xi = reshape_pt1(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    Gamma2 = reshape_pt1([k3 * g ** 3])
    if LS_deriv:
        mean_deriv = LS_deriv(reshape_pt1(xhat), reshape_pt1(u),
                              kwargs.get('prior_kwargs'))
    else:
        mean_deriv = np.zeros_like(xhat)
    if np.any(kwargs.get('saturation')):
        # Saturate the derivative of the nonlinearity estimate to guarantee
        # contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean_deriv = np.clip(mean_deriv, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(xi))
    DfA = reshape_pt1(np.dot(reshape_pt1_tonormal(mean_deriv),
                             reshape_pt1_tonormal(ABmult + u)))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    LC2 = reshape_pt1(Gamma2 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)
    xidot = reshape_pt1(DfA + LC2)
    # Also check eigenvalues of M for stability without high gain
    AB = np.concatenate((A, B), axis=1)
    ABO = np.concatenate((AB, np.zeros_like(reshape_pt1(AB[0]))), axis=0)
    K = np.array([[k1, k2, k3]])
    C = np.zeros_like(x)
    C[0, 0] = 1
    M = ABO - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return np.concatenate((xhatdot, xidot), axis=1)


# Observer for the continuous time Van der Pol equation
# Source: Inspired by Observer design for the duffing equation using
# Gersgorin’s theorem, by Alberto Delgado, using the approximation x**2v -
# xhat**2vhat approximately 2xhatvhat (x-xhat)
def VanderPol_observer_simplified(t, xhat, u, y, t0, init_control, GP, kwargs):
    mu = kwargs.get('mu')
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # My gains
    l1 = mu - 5
    # l2 = 1 - mu ** 2 - 2 * mu * xhat[:, 0] * xhat[:, 1]
    l2 = 1 - mu ** 2 - 2 * mu * xhat[:, 0]
    A = reshape_pt1([[0, 1], [-1, 0]])
    F = reshape_pt1([0, mu * (1 - xhat[:, 0] ** 2) * xhat[:, 1]])
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(xhat)) + F + LC + u)
    return xhatdot


# Linear Luenberger observer for harmonic oscillator, with control law u(t),
# continuous time
def harmonic_oscillator_observer_GP(t, xhat, u, y, t0, init_control, GP,
                                    kwargs):
    k = kwargs.get('k')
    m = kwargs.get('m')
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    # Gains
    l1 = - 1
    l2 = k / m - 1
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # In this case we have continuous observer dynamics, but GP is
                # discrete # TODO better than Euler?
                mean = (mean - xhat) / GP.prior_kwargs.get('dt')
            GP.prior_kwargs['observer_gains'] = {'l1': l1, 'l2': l2}
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u),
                      kwargs.get('prior_kwargs'))
    else:
        mean = np.zeros_like(u)
    LC = reshape_pt1([[l1 * (xhat[:, 0] - y), l2 * (xhat[:, 0] - y)]])
    xhatdot = reshape_pt1(mean + LC)
    return xhatdot


# High gain extended observer from Michelangelo for the WDC data
# Using current GP estimation of dynamics for xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
def WDC_observer_Michelangelo_GP(t, xhat, u, y, t0, init_control, GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    xi = reshape_pt1(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    Gamma2 = reshape_pt1([k3 * g ** 3])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean_deriv, var_deriv, lowconf_deriv, uppconf_deriv = \
                GP.predict_deriv(reshape_pt1(xhat), reshape_pt1(u), only_x=True)
            GP.prior_kwargs['observer_gains'].update({'g': g, 'Gamma1': Gamma1,
                                                      'Gamma2': Gamma2,
                                                      'k1': k1, 'k2': k2,
                                                      'k3': k3})
        else:
            mean_deriv = GP(reshape_pt1(xhat), reshape_pt1(u),
                            kwargs.get('prior_kwargs'))
    else:
        mean_deriv = np.zeros_like(xhat)
    if np.any(kwargs.get('saturation')):
        # Saturate the derivative of the nonlinearity estimate to guarantee
        # contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean_deriv = np.clip(mean_deriv, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(xi))
    DfA = reshape_pt1(np.dot(reshape_pt1_tonormal(mean_deriv),
                             reshape_pt1_tonormal(ABmult + u)))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    LC2 = reshape_pt1(Gamma2 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)
    xidot = reshape_pt1(DfA + LC2)

    # Also check eigenvalues of M for stability without high gain
    AB = np.concatenate((A, B), axis=1)
    ABO = np.concatenate((AB, np.zeros_like(reshape_pt1(AB[0]))), axis=0)
    K = np.array([[k1, k2, k3]])
    C = np.zeros_like(x)
    C[0, 0] = 1
    M = ABO - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return np.concatenate((xhatdot, xidot), axis=1)


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current GP estimation of dynamics for xdot, regular high gain observer
def WDC_observer_highgain_GP(t, xhat, u, y, t0, init_control, GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # discrete model so need to differentiate it in continuous obs
                mean = (mean - xhat) / GP.prior_kwargs.get(
                    'dt')  # TODO better than Euler?
            GP.prior_kwargs['observer_gains'].update({'g': g, 'Gamma1': Gamma1,
                                                      'k1': k1, 'k2': k2})
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                'prior_kwargs'))
    else:
        mean = np.zeros_like(xhat)
    if np.any(kwargs.get('saturation')):
        # Saturate the derivative of the nonlinearity estimate to guarantee
        # contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(mean + LC1 + u)
    return reshape_pt1(xhatdot)


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity
def WDC_justvelocity_observer_highgain_GP(t, xhat, u, y, t0, init_control,
                                          GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # discrete model so need to differentiate it in continuous obs
                mean = (mean - reshape_pt1(xhat[:, 1])) / GP.prior_kwargs.get(
                    'dt')  # TODO better than Euler?
            GP.prior_kwargs['observer_gains'].update({'g': g, 'Gamma1': Gamma1,
                                                      'k1': k1, 'k2': k2})
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                'prior_kwargs'))
    else:
        mean = np.zeros_like(reshape_pt1(xhat[:, 1]))
    if np.any(kwargs.get('saturation')):
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(mean))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)

    # Also check eigenvalues of M for stability without high gain
    K = np.array([[k1, k2]])
    C = np.zeros_like(xhat)
    C[0, 0] = 1
    M = A - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return reshape_pt1(xhatdot)


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current LS estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity
def WDC_justvelocity_observer_highgain_LS(t, xhat, u, y, t0, init_control,
                                          LS, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    dt = kwargs.get('dt')
    if LS:
        mean = LS(reshape_pt1(xhat), reshape_pt1(u), kwargs.get('prior_kwargs'))
        if not kwargs.get('continuous_model'):
            # discrete model so need to differentiate it in continuous obs
            mean = (mean - reshape_pt1(xhat[:, 1])) / kwargs.get('dt')
            # TODO better than Euler?
    else:
        mean = np.zeros_like(reshape_pt1(xhat[:, 1]))
    if np.any(kwargs.get('saturation')):
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(mean))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)

    # Also check eigenvalues of M for stability without high gain
    K = np.array([[k1, k2]])
    C = np.zeros_like(xhat)
    C[0, 0] = 1
    M = A - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return reshape_pt1(xhatdot)


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity, but returning xnext using
# Euler discretization instead of xdot
def WDC_justvelocity_discrete_observer_highgain_GP(t, xhat, u, y, t0,
                                                   init_control, GP, kwargs):
    xhatdot = WDC_justvelocity_observer_highgain_GP(t, xhat, u, y, t0,
                                                    init_control, GP, kwargs)
    xnext = reshape_pt1(xhat + kwargs.get('dt_before_subsampling') * xhatdot)
    # TODO better than Euler?
    return xnext


# High gain observer (simple, not extended like Michelangelo) for the WDC data
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity and with gain following a
# dynamical adaptation law
def WDC_justvelocity_observer_adaptive_highgain_GP(t, xhat, u, y, t0,
                                                   init_control, GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    g = float(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    adaptation_law = \
        kwargs.get('prior_kwargs').get('observer_gains').get('adaptation_law')

    # Gain (needs to be large enough)
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # discrete model so need to differentiate it in continuous obs
                mean = (mean - reshape_pt1(xhat[:, 1])) / GP.prior_kwargs.get(
                    'dt')  # TODO better than Euler?
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                'prior_kwargs'))
    else:
        mean = np.zeros_like(reshape_pt1(xhat[:, 1]))
    if np.any(kwargs.get('saturation')):
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        a_max = np.max([-kwargs.get('saturation'), kwargs.get('saturation')],
                       axis=0)
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1], [0, 0]])
    B = reshape_pt1([[0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(mean))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1 + u)
    gdot = reshape_pt1(adaptation_law(g=g, y=y, yhat=reshape_pt1(xhat[:, 0]),
                                      kwargs=kwargs.get('prior_kwargs').get(
                                          'observer_gains')))

    # Also check eigenvalues of M for stability without high gain
    K = np.array([[k1, k2]])
    C = np.zeros_like(xhat)
    C[0, 0] = 1
    M = A - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return reshape_pt1(np.concatenate((xhatdot, gdot), axis=1))


# High gain extended observer from Michelangelo for the mass-spring-mass system
# Using current GP estimation of dynamics for xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
def MSM_observer_Michelangelo_GP(t, xhat, u, y, t0, init_control, GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    xi = reshape_pt1(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    k4 = kwargs.get('prior_kwargs').get('observer_gains').get('k4')
    k5 = kwargs.get('prior_kwargs').get('observer_gains').get('k5')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2, k3 * g ** 3, k4 * g ** 4])
    Gamma2 = reshape_pt1([k5 * g ** 5])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean_deriv, var_deriv, lowconf_deriv, uppconf_deriv = \
                GP.predict_deriv(reshape_pt1(xhat), reshape_pt1(u), only_x=True)
        else:
            mean_deriv = GP(reshape_pt1(xhat), reshape_pt1(u),
                            kwargs.get('prior_kwargs'))
    else:
        mean_deriv = np.zeros_like(xhat)
    if np.any(kwargs.get('saturation')):
        # Saturate the derivative of the nonlinearity estimate to guarantee
        # contraction
        a_min = np.min([-kwargs.get('saturation')])
        a_max = np.max([kwargs.get('saturation')])
        mean_deriv = np.clip(mean_deriv, a_min=a_min, a_max=a_max)
    A = np.eye(xhat.shape[1], k=1)
    B = np.zeros((xhat.shape[1], 1))
    B[-1] = 1
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(xi))
    DfA = reshape_pt1(np.dot(reshape_pt1_tonormal(mean_deriv),
                             reshape_pt1_tonormal(ABmult)))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    LC2 = reshape_pt1(Gamma2 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1)
    xidot = reshape_pt1(DfA + LC2)

    # Also check eigenvalues of M for stability without high gain
    AB = np.concatenate((A, B), axis=1)
    ABO = np.concatenate((AB, np.zeros_like(reshape_pt1(AB[0]))), axis=0)
    K = np.array([[k1, k2, k3, k4, k5]])
    C = np.zeros_like(x)
    C[0, 0] = 1
    M = ABO - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return np.concatenate((xhatdot, xidot), axis=1)


# High gain observer for the mass-spring-mass system
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity
def MSM_justvelocity_observer_highgain_GP(t, xhat, u, y, t0, init_control,
                                          GP, kwargs):
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(xhat)
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))

    # Gain (needs to be large enough)
    g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    k4 = kwargs.get('prior_kwargs').get('observer_gains').get('k4')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2, k3 * g ** 3, k4 * g ** 4])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # discrete model so need to differentiate it in continuous obs
                mean = (mean - reshape_pt1(xhat[:, -1])) / GP.prior_kwargs.get(
                    'dt')  # TODO better than Euler?
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                'prior_kwargs'))
    else:
        mean = np.zeros_like(reshape_pt1(xhat[:, -1]))
    if np.any(kwargs.get('saturation')):
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([kwargs.get('saturation')])
        a_max = np.max([kwargs.get('saturation')])
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    B = reshape_pt1([[0], [0], [0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(mean))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1)

    # Also check eigenvalues of M for stability without high gain
    K = np.array([[k1, k2, k3, k4]])
    C = np.zeros_like(xhat)
    C[0, 0] = 1
    M = A - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return reshape_pt1(xhatdot)


# High gain observer for the mass-spring-mass system
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity and with gain following a
# dynamical adaptation law
def MSM_justvelocity_observer_adaptive_highgain_GP(t, xhat, u, y, t0,
                                                   init_control, GP, kwargs):
    x = reshape_pt1(xhat)
    assert np.any(kwargs.get('saturation')), 'Need to define a saturation ' \
                                             'value to use the combined ' \
                                             'observer-identifier framework.'
    xhat = reshape_pt1(x[:, :-1])
    g = float(x[:, -1])
    y = reshape_pt1(y(t, kwargs))
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    adaptation_law = \
        kwargs.get('prior_kwargs').get('observer_gains').get('adaptation_law')

    # Gain (needs to be large enough)
    k1 = kwargs.get('prior_kwargs').get('observer_gains').get('k1')
    k2 = kwargs.get('prior_kwargs').get('observer_gains').get('k2')
    k3 = kwargs.get('prior_kwargs').get('observer_gains').get('k3')
    k4 = kwargs.get('prior_kwargs').get('observer_gains').get('k4')
    Gamma1 = reshape_pt1([k1 * g, k2 * g ** 2, k3 * g ** 3, k4 * g ** 4])
    if GP:
        if 'GP' in GP.__class__.__name__:
            mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                     reshape_pt1(u))
            if not kwargs.get('continuous_model'):
                # discrete model so need to differentiate it in continuous obs
                mean = (mean - reshape_pt1(xhat[:, -1])) / GP.prior_kwargs.get(
                    'dt')  # TODO better than Euler?
        else:
            mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                'prior_kwargs'))
    else:
        mean = np.zeros_like(reshape_pt1(xhat[:, -1]))
    if np.any(kwargs.get('saturation')):
        # Saturate the estimate of the nonlinearity to guarantee contraction
        a_min = np.min([-kwargs.get('saturation')])
        a_max = np.max([kwargs.get('saturation')])
        mean = np.clip(mean, a_min=a_min, a_max=a_max)
    A = reshape_pt1([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    B = reshape_pt1([[0], [0], [0], [1]])
    ABmult = np.dot(A, reshape_pt1_tonormal(xhat)) + \
             np.dot(B, reshape_pt1_tonormal(mean))
    LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
    xhatdot = reshape_pt1(ABmult + LC1)
    gdot = reshape_pt1(adaptation_law(g=g, y=y, yhat=reshape_pt1(xhat[:, 0]),
                                      kwargs=kwargs.get('prior_kwargs').get(
                                          'observer_gains')))

    # Also check eigenvalues of M for stability without high gain
    K = np.array([[k1, k2, k3, k4]])
    C = np.zeros_like(xhat)
    C[0, 0] = 1
    M = A - np.dot(K.T, C)
    eigvals = np.linalg.eigvals(M)
    for x in eigvals:
        if np.linalg.norm(np.real(x)) < 1e-5:
            logging.warning('The eigenvalues of the matrix M are dangerously '
                            'small, low robustness of the observer! Increase '
                            'the gains.')
        elif np.real(x) > 0:
            logging.warning('Some of the eigenvalues of the matrix M are '
                            'positive. Change the gains to get a Hurwitz '
                            'matrix.')
    return reshape_pt1(np.concatenate((xhatdot, gdot), axis=1))


# Functions for observing experimental data from full data
def dim1_observe_data(xtraj):
    return reshape_dim1(xtraj[:, 0])
