import subprocess
import wandb

sweep_configuration = {
    # "method": "random",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "mean"},
    "parameters": {
        "n": {"values": [5000]},
        "d": {"values": [20]},
        "r": {"values": [3]},
        "n_envs": {"values": [10]},
        "n_test_envs": {"values": [20]},
        "epsilon": {"values": [0.1]},
        "gamma": {"values": [0.98]},
        "activation": {"values": ["mixed"]},
        "steps": {"values": [50000]},
        "estimator": {"values": ["linear"]},
        "model_activation": {"values": ["sigmoid", "learned"]},
        "model_hidden_size": {"values": [4]},
        "bandwidth": {"values": [7.0]},
        "reg": {"values": [0, 1e-5, 1e-4]},
        "mono": {"values": [False]},
        "n_samples_burnin": {"values": [500]},
        "intv_scale": {"values": [0.3]},
        "learning_rate": {"values": [0.003]},
        "weight_decay": {"values": [0.0001]},
        "seed": {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
        "val_metric": {"values": ["mean"]}, #"w2"
        "scale": {"values": [1e-1]},
    },
}

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="fresh-jax-sweep")
    def run_sweep():
        subprocess.run(["python", "nonlinear_script.py"], check=True)  
    wandb.agent(sweep_id, function=run_sweep)
