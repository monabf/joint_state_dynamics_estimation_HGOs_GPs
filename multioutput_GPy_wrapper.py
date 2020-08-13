import GPy
import numpy as np

from utils import reshape_dim1, reshape_pt1, reshape_pt1_tonormal


class MultiOutput_GPy_Wrapper:

    # Wraps the usual GPy functions (model, predict, optimize...) to apply
    # them on a list of GPs, one for each output dimension

    def __init__(self, X, Y, kernel, likelihood, inference_method):
        self.nb_output_dims = reshape_pt1(Y[0]).shape[1]
        self.kern = kernel
        self.models = [GPy.core.gp.GP(X, reshape_dim1(Y[:, 0]),
                                      kernel=self.kern[0],
                                      likelihood=likelihood,
                                      inference_method=inference_method)]
        self.param_array = np.array(self.models[0].param_array)
        for i in range(1, self.nb_output_dims):
            self.models += [
                GPy.core.gp.GP(X, reshape_dim1(Y[:, i]), self.kern[i],
                               likelihood,
                               inference_method=inference_method)]
            self.param_array = np.concatenate((self.param_array,
                                               self.models[i].param_array))

    def __str__(self):
        model_strings = ''
        for i in range(self.nb_output_dims):
            model_strings += self.models[i].__str__()
            model_strings += '\n'
        return model_strings

    def __repr__(self):
        model_strings = ''
        for i in range(self.nb_output_dims):
            model_strings += self.models[i].__str__()
            model_strings += '\n'
        return model_strings

    def create_model(self, X, Y, likelihood, inference_method):
        self.models = [GPy.core.gp.GP(X, reshape_dim1(Y[:, 0]), self.kern[0],
                                      likelihood,
                                      inference_method=inference_method)]
        self.param_array = np.array(self.models[0].param_array)
        for i in range(1, self.nb_output_dims):
            self.models += [
                GPy.core.gp.GP(X, reshape_dim1(Y[:, i]), self.kern[i],
                               likelihood,
                               inference_method=inference_method)]
            self.param_array = np.concatenate((self.param_array,
                                               self.models[i].param_array))
        return self.models

    def optimize_restarts(self, num_restarts=5, verbose=False, max_iters=100,
                          robust=True):
        for i in range(self.nb_output_dims):
            self.models[i].optimize_restarts(num_restarts=num_restarts,
                                             verbose=verbose,
                                             max_iters=max_iters, robust=robust)
            self.kern[i] = self.models[i].kern
        self.param_array = np.array(self.models[0].param_array)
        for i in range(1, self.nb_output_dims):
            self.param_array = np.concatenate((self.param_array,
                                               self.models[i].param_array))
        for i in range(self.nb_output_dims):
            print(self.kern[i])

    def predict(self, X, full_cov=False):
        means = reshape_pt1(
            np.array(self.models[0].predict(X, full_cov=full_cov)[0]))
        covars = reshape_pt1(
            np.array(self.models[0].predict(X, full_cov=full_cov)[1]))
        for i in range(1, self.nb_output_dims):
            means = np.concatenate((means, reshape_pt1(
                self.models[i].predict(X, full_cov=full_cov)[0])), axis=1)
            covars = np.concatenate((covars, reshape_pt1(
                self.models[i].predict(X, full_cov=full_cov)[1])), axis=1)
        assert means.shape[1] == self.nb_output_dims, \
            'Wrong shapes in multioutput GP predicted mean'
        assert covars.shape[1] == self.nb_output_dims, \
            'Wrong shapes in multioutput GP predicted covar'
        var = reshape_pt1(np.linalg.det(np.diag(reshape_pt1_tonormal(
            covars)))) ** (1 / self.nb_output_dims)
        return reshape_pt1(means), var