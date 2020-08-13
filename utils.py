import logging

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal


# Some useful functions


# Root mean square of a matrix
def RMS(x):
    return np.sqrt(np.sum(np.mean(np.square(x), axis=0)))


# Log likelihood of a matrix given a mean and variance of same shape
def log_multivariate_normal_likelihood(x, mean, var):
    assert x.shape == mean.shape, 'Data and mean do not have the same shape'
    log_likelihood_array = np.zeros((x.shape[0], 1))
    for idx, xi in enumerate(x):
        if reshape_pt1_tonormal(var[idx]).shape[0] == 1:
            covar = float(reshape_pt1_tonormal(var[idx]))
        else:
            covar = reshape_pt1_tonormal(var[idx])
        if np.array(covar).all() == 0:
            covar = 1e-8 * np.ones_like(covar)
        log_likelihood_array[idx] = reshape_pt1(
            multivariate_normal.logpdf(xi, mean=mean[idx], cov=covar))
    log_likelihood = float(np.mean(log_likelihood_array, axis=0))
    return log_likelihood


# Reshape any vector of (length,) object to (length, 1) (possibly several
# points but of dimension 1)
def reshape_dim1(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1, 1))
    else:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    if verbose:
        print(x.shape)
    return x


# Reshape any vector of (length,) object to (1, length) (single point of
# certain dimension)
def reshape_pt1(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1, 1))
    else:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (1, x.shape[0]))
    if verbose:
        print(x.shape)
    return x


# Reshape any point of type (1, length) to (length,)
def reshape_pt1_tonormal(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1,))
    elif len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0],))
    elif x.shape[0] == 1:
        x = np.reshape(x, (x.shape[1],))
    if verbose:
        print(x.shape)
    return x


# Reshape any vector of type (length, 1) to (length,)
def reshape_dim1_tonormal(x, verbose=False):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1,))
    elif len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0],))
    elif x.shape[1] == 1:
        x = np.reshape(x, (x.shape[0],))
    if verbose:
        print(x.shape)
    return x


# Functions returning the value of the information criterion to optimize at a
# certain point, given a trained GP model
def posterior_variance(x, model):
    x = reshape_pt1(x, verbose=False)
    (mean, var) = model.predict(x)
    return var


def entropy(x, model):
    x = reshape_pt1(x, verbose=False)
    (mean, var) = model.predict(x)
    return 1 / 2 * np.log(2 * np.pi * np.exp(0) * var ** 2)


# Remove outliers from a pandas dataframe
def remove_outlier(df):
    # https://gist.github.com/ariffyasri/70f1e9139da770cb8514998124560281
    low = .001
    high = .999
    quant_df = df.quantile([low, high])
    mask = [True]
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            mask = (df[name] >= quant_df.loc[low, name]) & (
                    df[name] <= quant_df.loc[high, name])
    return mask


# Vector x = (t, x) of time steps t at which x is known is interpolated at given
# time t, imposing initial value, and interpolating along each output dimension
# independently if there are more than one
def interpolate(t, x, t0, init_value, method='cubic'):
    x = reshape_pt1(x)
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = reshape_dim1_tonormal(t)
    tf = x[-1, 0]

    if len(x) == 1:
        # If only one value of x available, assume constant
        interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
    else:
        # Interpolate data t_x at array of times wanted; if several output
        # dims, interpolate all input dims for each output dim
        interpolate_x = griddata(x[:, 0], x[:, 1:], t, method=method)

    if t[0] == t0:
        # Impose initial value
        interpolate_x[0] = reshape_pt1(init_value)

    # Interpolation slightly outside of range
    if len(x) >= 2:
        tol = 100 * (tf - x[-2, 0])
        if tf < t[-1] <= tf + tol:
            # If t[-1] less than tol over last available t, return x[-1]
            interpolate_x[-1] = reshape_pt1(x[-1, 1:])
        elif t0 > t[0] >= t0 - tol:
            # If t[0] lass than tol before first available t, return x[0]
            interpolate_x[0] = reshape_pt1(init_value)

    if np.isnan(np.min(interpolate_x)):
        print(t, x)
        logging.error('NaNs in interpolation: values need to be interpolated '
                      'outside of range, should not happen!')
    return reshape_pt1(interpolate_x)
