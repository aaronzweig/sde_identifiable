import subprocess
import wandb

sweep_configuration = {
    # "method": "random",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "mean"},
    "parameters": {
        "n": {"values": [500]},
        "d": {"values": [20]},
        "r": {"values": [2]},
        "n_envs": {"values": [10]},
        "n_test_envs": {"values": [20]},
        "epsilon": {"values": [0.5]},
        "gamma": {"values": [0.98]},
        "activation": {"values": ["mixed"]},
        "steps": {"values": [30000]},
        "estimator": {"values": ["linear"]},
        "model_activation": {"values": ["sigmoid", "learned"]},
        "model_hidden_size": {"values": [4, 8, 16]},
        "bandwidth": {"values": [5.0, 7.0]},
        "reg": {"values": [0, 0.0001, 0.001]},
        "n_samples_burnin": {"values": [200]},
        "intv_scale": {"values": [0.1]},
        "learning_rate": {"values": [0.003]},
        "weight_decay": {"values": [0.0001]},
        "seed": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    },
}

# sweep_configuration = {
#     # "method": "random",
#     "method": "grid",
#     "metric": {"goal": "minimize", "name": "mean"},
#     "parameters": {
#         "n": {"values": [2000]},
#         "d": {"values": [5, 20]},
#         "r": {"values": [4]},
#         "n_envs": {"values": [10]},
#         "n_test_envs": {"values": [20]},
#         "epsilon": {"values": [1.0, 1.5]},
#         "gamma": {"values": [0.98]},
#         "activation": {"values": ["cursed"]},
#         "steps": {"values": [20000]},
#         "estimator": {"values": ["linear"]},
#         "model_activation": {"values": ["sigmoid", "learned"]},
#         "model_hidden_size": {"values": [4, 8]},
#         "bandwidth": {"values": [5.0, 7.0]},
#         "reg": {"values": [0.0, 0.0001, 0.001]},
#         "n_samples_burnin": {"values": [200]},
#         "intv_scale": {"values": [0.1]},
#         "seed": {"values": [1,2,3,4,5]}
#     },
# }

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="motivated-jax-sweep")
    def run_sweep():
        subprocess.run(["python", "nonlinear_script.py"], check=True)  
    wandb.agent(sweep_id, function=run_sweep)
