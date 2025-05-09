import sys
import os
sys.path.append(os.path.abspath(".."))
from jax import random, numpy as jnp
from stadion.models import LinearSDE, MLPSDE
from stadion.kds import kds_loss
from pprint import pprint
from stadion import kds_loss
from stadion.inference import rbf_kernel
from scipy.linalg import solve_continuous_lyapunov
import numpy as onp
import ot
import jax
from functools import partial


from nn import NNSDE

import wandb

def sample_stiefel(key, n, m):
    assert m <= n
    X = random.normal(key, shape = (n, m))
    Z = X @ jnp.linalg.inv(jax.scipy.linalg.sqrtm(X.T @ X))
    return jnp.real(Z)

def build_model(hidden_size, activation, epsilon, gamma, n_samples_burnin, mono):
    model = NNSDE(
        hidden_size = hidden_size, 
        activation=activation,
        epsilon = epsilon,
        gamma = gamma,
        mono = mono,
        sde_kwargs = {"n_samples_burnin": n_samples_burnin})
    return model

def initialize_model(key, model, d):
    model.n_vars = d
    r = model.hidden_size
    
    key, subk = random.split(key)
    param = model.init_param(subk, d, scale=1.0)
    
    param["mlp_0"] = jax.numpy.eye(d, r)
    param["mlp_1"] = jax.numpy.eye(r, d)
    
    # key, subk1, subk2 = random.split(key, 3)
    # param["mlp_0"] = sample_stiefel(subk1, d, r)
    # param["mlp_1"] = sample_stiefel(subk2, d, r).T
    
    param["mlp_b_0"] *= 0
    param["mlp_b_1"] *= 0
    model.param = param
    return key, model, param

def build_data(key, model, d, n_envs, intv_scale, n):
    targets = [jnp.ones(d) for _ in range(n_envs)]
    key, subk = random.split(key)
    intv_param = model.init_intv_param(subk, d = d, scale=intv_scale, n_envs = n_envs, targets = targets)

    datasets = []
    for k in range(n_envs):
        local_intv_param = intv_param.index_at(k)
        key, subk = random.split(key)
        data = model.sample(subk, n_samples = n, intv_param = local_intv_param)
        datasets.append(data)

    return key, intv_param, datasets, targets
    

def fit_model(key, model, datasets, targets, intv_param, bandwidth, steps, estimator, reg, learning_rate, weight_decay, scale):

    key, subk = random.split(key)
    
    model.fit(
        subk,
        x=datasets,
        intv_param=intv_param,
        targets=targets,
        bandwidth = bandwidth,
        steps = steps,
        estimator=estimator,
        reg = reg,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scale=scale,
    )

    return key, model

def predict_datasets(key, model, n_envs, intv_param, n):
    pred_datasets = []
    for k in range(n_envs):
        local_intv_param = intv_param.index_at(k)
        key, subk = random.split(key)
        data = model.sample(subk, n_samples = n, intv_param = local_intv_param)
        pred_datasets.append(data)
    return pred_datasets

def assess_model_mean(key, model, n_envs, intv_param, datasets, n):
    pred_datasets = predict_datasets(key, model, n_envs, intv_param, n)

    X = jnp.mean(jnp.stack(pred_datasets), axis = 1)
    Y = jnp.mean(jnp.stack(datasets), axis = 1)
    norms = jnp.linalg.norm(X - Y, axis = 1)
    return norms, jnp.mean(norms), jnp.std(norms)

def assess_model_ksd(key, model, n_envs, intv_param, datasets, n, bandwidth):
    # pred_datasets = predict_datasets(key, model, n_envs, intv_param, n)

    if isinstance(bandwidth, list):
        kernel = lambda x, y: sum([partial(rbf_kernel, bandwidth=float(b))(x, y) for b in bandwidth])
    else:
        kernel = partial(rbf_kernel, bandwidth=float(bandwidth))

    loss_fun = kds_loss(model.f, model.sigma, kernel, estimator="u-statistic")

    ksds = []
    for k in range(n_envs):
        Y = datasets[k]
        ksd = loss_fun(Y, model.param, intv_param.index_at(k))
        ksds.append(ksd)
    ksds = jnp.array(ksds)
    return ksds, jnp.mean(ksds), jnp.std(ksds)
    
def assess_model(key, model, n_envs, intv_param, datasets, n, ot_epsilon = 0.1):
    pred_datasets = predict_datasets(key, model, n_envs, intv_param, n)

    wds = []
    a = 1.0 / n * jnp.ones(n)
    for k in range(n_envs):
        X = pred_datasets[k]
        Y = datasets[k]
        M = ot.dist(X, Y)
        dist = ot.sinkhorn2(a, a, M, ot_epsilon, method="sinkhorn_log")
        wds.append(dist)
    wds = jnp.array(wds)
    if jnp.std(wds) < 1e-4: #OT failed
        return wds, jnp.array(100.0), jnp.array(100.0)
    return wds, jnp.mean(wds), jnp.std(wds)

def run_model():
    wandb.init()
    config = wandb.config
    key = random.PRNGKey(config.seed)

    true_model = build_model(config.r, config.activation, config.epsilon, config.gamma, config.n_samples_burnin, mono=False)
    key, true_model, param = initialize_model(key, true_model, config.d)
    
    key, intv_param, datasets, targets = build_data(key, true_model, config.d, config.n_envs, config.intv_scale, config.n)
    key, test_intv_param, test_datasets, test_targets = build_data(key, true_model, config.d, config.n_test_envs, config.intv_scale, config.n)


    model = build_model(config.model_hidden_size, config.model_activation, config.epsilon, config.gamma, config.n_samples_burnin, config.mono)
    key, model = fit_model(key, model, datasets, targets, intv_param, config.bandwidth, config.steps, config.estimator, config.reg, config.learning_rate, config.weight_decay, config.scale)

    if config.val_metric == "mean":
        wds, mean, std = assess_model_mean(key, model, config.n_test_envs, test_intv_param, test_datasets, config.n)
    elif config.val_metric == "w2":
        wds, mean, std = assess_model(key, model, config.n_test_envs, test_intv_param, test_datasets, config.n, ot_epsilon = 0.2)
    else:
        raise ValueError(f"Unknown metric: {config.metric}")
    wandb.log({"mean": mean})
    wandb.log({"std": std})
    wandb.finish()
    


if __name__ == "__main__":
    run_model()
