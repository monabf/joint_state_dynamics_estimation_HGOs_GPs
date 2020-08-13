import logging

import seaborn as sb

sb.set_style('whitegrid')

# Class for efficiently handling configurations and parameters, enables to
# easily set them and remember them when one config is reused

default_config = dict(true_meas_noise_var=0,
                      process_noise_var=0,
                      optim_method='RK45',
                      nb_rollouts=0,
                      nb_loops=1,
                      rollout_length=100,
                      sliding_window_size=None,
                      verbose=False,
                      monitor_experiment=True,
                      multioutput_GP=False,
                      sparse=None,
                      memory_saving=False,
                      restart_on_loop=False,
                      GP_optim_method='lbfgsb',
                      meas_noise_var=0.1,
                      batch_adaptive_gain=None,
                      nb_plotting_pts=500,
                      no_control=False,
                      full_rollouts=False,
                      max_rollout_value=500)


class Config:

    def __init__(self, **kwargs):
        self.params = kwargs

        # Check that necessary keys have been filled in
        mandatory_keys = ['system', 'nb_samples', 't0_span', 'tf_span', 't0',
                          'tf', 'hyperparam_optim']
        for key in mandatory_keys:
            assert key in self.params, 'Mandatory key ' + key \
                                       + ' was not given.'
        self.params['dt'] = (self.tf - self.t0) / self.nb_samples
        if 'Continuous_model' in self.params['system']:
            self.params['continuous_model'] = True
        else:
            self.params['continuous_model'] = False

        # Fill other keys with default values
        for key in default_config:
            if key not in self.params:
                self.params[key] = default_config[key]
        if 'rollout_controller' not in self.params:
            self.params['rollout_controller'] = \
                {'random': self.params['nb_rollouts']}

        # Warn / assert for specific points
        if self.t0 != 0 or not self.restart_on_loop:
            logging.warning(
                'Initial simulation time is not 0 for each scenario! This is '
                'incompatible with DynaROM.')
        assert not (self.batch_adaptive_gain and ('adaptive' in self.system)), \
            'Cannot adapt the gain both through a continuous dynamic and a ' \
            'batch adaptation law.'

        # Check same number of rollouts as indicated in rollout_controller
        nb_rollout_controllers = 0
        for key, val in self.params['rollout_controller'].items():
            nb_rollout_controllers += val
        assert nb_rollout_controllers == self.params['nb_rollouts'], \
            'The number of rollouts given by nb_rollouts and ' \
            'rollout_controller should match.'

    def __getattr__(self, item):
        # self.params[item] can be called directly as self.item
        return self.params[item]

    def __iter__(self):
        # Iterating through self is the same as iterating through self.params
        return iter(self.params)

    def __next__(self):
        # Iterating through self is the same as iterating through self.params
        return next(self.params)

    def __getitem__(self, item):
        # self[item] means self.params[item]
        return self.params.__getitem__(item)

    def __str__(self):
        # print(self) is same as print(self.params)
        return self.params.__str__()

    def update(self, dict):
        # Updating self means updating self.params
        self.params.update(dict)

    def get(self, item):
        # Function get can also be used same as __getitem__
        return self.params.get(item)

    def copy(self):
        # Make a new object with the same parameters
        new_copy = Config(**self.params)
        return new_copy

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for key, val in self.params.items():
                print(key, ': ', val, file=f)


class Test:

    def __init__(self, config: Config):
        self.a = 0
        self.config = config

    def __getattr__(self, item):
        return self.config.__getattr__(item)


if __name__ == '__main__':
    config = Config(system='Continuous/Louise_example/Basic_Louise_case',
                    nb_samples=int(1e4),
                    t0_span=0,
                    tf_span=int(1e2),
                    t0=0,
                    tf=int(1e2),
                    hyperparam_optim='fixed_hyperparameters')
    test = Test(config)
    print(test.config, test.config.t0, config.t0)
    print('Test keys:')
    for key in test.config:
        print(key, test.config.params[key])
