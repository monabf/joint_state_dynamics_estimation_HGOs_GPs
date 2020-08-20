Code for paper "Joint state and dynamics estimation with high-gain observers and Gaussian process models", by Mona Buisson-Fenet, Valéry
  Morgenthaler, Sebastian Trimpe, and Florent Di Meglio. Focusing on a dynamical
   system in observable canonical form with an unknown nonlinearity, the aim
    is to jointly estimate the state with a high-gain observer and the nonlinearity with a Gaussian process model. 
 \
 \
 To reproduce the results from the paper:
 - create a directory for this repo, further named dir
 - create dir/Figures/Logs
 - clone this repo in dir/repo
 - create a python3 virtual environment for this repo in dir/venv, source it
 - install all requirements with pip (pip install -r repo/requirements.txt)
 - if any problems occur during the installation of required packages, see requirements.txt for possible fixes
 - if you want to use multivariate Gaussian priors for hyperparameter
  optimization, replace the file priors.py in your virtual environment dir/venv/lib/python3.7/site-packages/GPy
 /core/parametrization/ by the file dir/repo/priors.py (pull request pending)
 - cd repo, run python mass_spring_mass_testcase/MSM_observer_GP.py 1 (1
  = process number, used for logging)

This will run the simulation described in the paper for 10 cycles, with 10
 rollouts used for evaluation at each cycle. Modify the config parameters in the
  script to change the settings.

To play with other methods and test cases, look into the
 quasilinear_observer_GP.py file and choose which system and method to use, then run python quasilinear_observer_GP.py 1
 
