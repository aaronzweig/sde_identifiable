import subprocess
import wandb



sweep_configuration = {
    # "method": "random",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "mean"},
    "parameters": {
        "n": {"values": [1000]},
        "d": {"values": [20]},
        "r": {"values": [5]},
        "n_envs": {"values": [10]},
        "n_test_envs": {"values": [30]},
        "epsilon": {"values": [1.5]},
        "gamma": {"values": [0.99]},
        "activation": {"values": ["cursed", "linear"]},
        "steps": {"values": [20000]},
        "estimator": {"values": ["linear"]},
        "model_activation": {"values": ["sigmoid", "learned"]},
        "model_hidden_size": {"values": [4, 8]},
        "bandwidth": {"values": [7.0]},
        "reg": {"values": [0, 0.0001, 0.001]},
        "n_samples_burnin": {"values": [200]},
        "intv_scale": {"values": [0.01]},
        "seed": {"values": [123,124]}
    },
}

if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="motivated-jax-sweep")
    def run_sweep():
        subprocess.run(["python", "nonlinear_script.py"], check=True)  
    wandb.agent(sweep_id, function=run_sweep)
