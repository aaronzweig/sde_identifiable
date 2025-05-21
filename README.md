# Towards Identifiability of Interventional Stochastic Differential Equations

This code reproduces the synthetic experiments in *"Towards Identifiability of Interventional Stochastic Differential Equations"*


The linear experiments can be reproduced with the scripts:
```bash

python sde_linear_script.py --n 100 --fix_decay True
python sde_linear_script.py --n 200 --fix_decay True
python sde_linear_script.py --n 100 --fix_decay False
python sde_linear_script.py --n 200 --fix_decay False
```

A sweep over hyperparameters for the nonlinear experiments can be reproduced with:
```bash
python wandb_script_multiseed.py
```

# Reference

TODO:
