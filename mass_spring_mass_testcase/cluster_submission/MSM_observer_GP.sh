# Copyright 2019 Max Planck Society. All rights reserved.
#!/bin/bash
source ../venv/bin/activate
log_nb=$(echo "scale=8; $1 +0.00001*$RANDOM +1" | bc)
python mass_spring_mass_testcase/MSM_observer_GP.py $log_nb
