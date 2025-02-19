import functools

import jax
from jax import vmap, random
import jax.numpy as jnp

from stadion.parameters import ModelParameters, InterventionParameters
from stadion.sde import SDE
from stadion.inference import KDSMixin
from stadion.utils import to_diag, tree_global_norm, tree_init_normal, tree_variance_initialization
from stadion.models import LinearSDE, MLPSDE

from monotonic import *



import time
from functools import partial
import math
import jax
from jax import numpy as jnp, random, tree_map
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from optax._src import linear_algebra
import optax

from stadion.kds import kds_loss
from stadion.data import make_dataloader
from stadion.utils import update_ave, retrieve_ave, tree_isnan
from stadion.inference import rbf_kernel
from collections import defaultdict



class NNSDE(MLPSDE):
    
    def __init__(
        self,
        sparsity_regularizer="both",
        hidden_size=8,
        activation="sigmoid",
        init_distribution="uniform",
        init_mode="fan_in",
        gamma=1.0,
        epsilon=1.0,
        sde_kwargs=None,
    ):

        sde_kwargs = sde_kwargs or {}
        SDE.__init__(self, **sde_kwargs)

        self.sparsity_regularizer = sparsity_regularizer

        cursed_act = lambda x: 1./3. * (jax.nn.leaky_relu(x, 0.1) + jax.nn.leaky_relu(x - 1, 0.2) + jax.nn.leaky_relu(x + 1, 0.3))

        if activation == "tanh":
            self.nonlin = jnp.tanh
        elif activation == "relu":
            self.nonlin = jax.nn.relu
        elif activation == "sigmoid":
            self.nonlin = jax.nn.sigmoid
        elif activation == "cursed":
            self.nonlin = cursed_act
        elif activation == "mixed":

            # For d=2, gives multiple modes
            # acts = [lambda x: 3*jnp.cos((x-0.5)*3), lambda x: 2*jnp.sin((x+1.5)*2) - 1]
            acts = [jax.nn.relu, cursed_act]
            def mixed_act(x, acts = acts):
                n = len(acts)

                result = jnp.empty_like(x)
                for i in range(n):
                    val = acts[i](x[..., i::n])
                    result = result.at[..., i::n].set(val)
                return result
                
            self.nonlin = lambda x: mixed_act(x)
        elif activation == "rbf":
            self.nonlin = lambda arr: jnp.exp(- jnp.power(arr, 2))
        elif activation == "linear":
            self.nonlin = lambda arr: arr
        elif activation == "learned":
            self.nonlin = None
        else:
            raise KeyError(f"Unknown activation {activation}")

        self.hidden_size = hidden_size
        self.init_distribution = init_distribution
        self.init_mode = init_mode
        self.epsilon = epsilon
        self.gamma = gamma


    def init_param(self, key, d, scale=1e-1, fix_speed_scaling=True):
        """
        Samples random initialization of the SDE model parameters.
        See :func:`~stadion.inference.KDSMixin.init_param`.
        """
        # layers should follow convention fan_in -> fan_out for proper initialization (taking into account vmap below)
        shape = {
            "mlp_0": jnp.zeros((d, self.hidden_size)),
            "mlp_b_0": jnp.zeros((self.hidden_size,)),
            "mlp_1": jnp.zeros((self.hidden_size, d)),
            # "mlp_b_1": jnp.zeros((d,)),
            "log_reversion": jnp.zeros((d,)),
        }

        if self.nonlin is None:
            self.mono = MonotonicMLP(input_dim = self.hidden_size, hidden_dim=100)
            shape = {**shape, **self.mono.shape}

        _initializer = functools.partial(tree_variance_initialization, scale=scale, mode=self.init_mode,
                                         distribution=self.init_distribution)

        param = _initializer(key, shape)
        param["mlp_0"] /= jnp.linalg.norm(param["mlp_0"], ord=2)
        param["mlp_1"] /= jnp.linalg.norm(param["mlp_1"], ord=2)
        param["log_reversion"] *= 0
        
        if fix_speed_scaling:
            return ModelParameters(
                parameters=param,
            )
        else:
            return ModelParameters(
                parameters=param,
            )

    def init_intv_param(self, key, d, n_envs=None, scale=1e-6, targets=None, x=None):
        """
        Samples random initialization of the intervention parameters.
        See :func:`~stadion.inference.KDSMixin.init_intv_param`.
        """
        # pytree of [n_envs, d, ...]
        # intervention effect parameters
        vec_shape = (n_envs, d) if n_envs is not None else (d,)
        shape = {
            "shift": jnp.zeros(vec_shape),
            "log_scale": jnp.zeros(vec_shape),
        }
        intv_param = tree_init_normal(key, shape, scale=scale)

        # if provided, store intervened variables for masking
        if targets is not None:
            targets = jnp.array(targets, dtype=jnp.float32)
            assert targets.shape == vec_shape

        return InterventionParameters(parameters=intv_param, targets=targets)

    """
    Model
    """

    def f(self, x, param, intv_param):

        if self.nonlin is None:
            act = functools.partial(self.mono.vmap_forward, param=param)
        else:
            act = self.nonlin

        z = self.gamma * act(x @ param["mlp_0"] + param["mlp_b_0"])
        x_out = z @ param["mlp_1"]
        # x_out += param["mlp_b_1"]
        f_vec = x_out - x @ jnp.diag(jnp.exp(param["log_reversion"]))
        # f_vec = x_out - x

        
        if intv_param is not None:
            f_vec += intv_param["shift"]

        assert x.shape == f_vec.shape
        return f_vec

    def sigma(self, x, param, intv_param):
        d = x.shape[-1]
        c = jnp.ones(d) * self.epsilon
        sig_mat = to_diag(jnp.ones_like(x)) * c
        return sig_mat


    """
    Inference functions
    """

    def modify_dparam(self, dparam):
        return dparam

    def regularize_sparsity(self, param):
        """
        Sparsity regularization.
        See :func:`~stadion.inference.KDSMixin.regularize_sparsity`.
        """

        l1 = lambda x: jnp.sum(jnp.abs(x))
        reg = l1(param["mlp_0"]) + l1(param["mlp_1"])
        if self.nonlin is None:
            reg += l1(param["A"]) + l1(param["B"])
        return reg


    

    def fit(
        self,
        key,
        x,
        intv_param, #Require to plant fixed interventions
        targets=None,
        bandwidth=5.0,
        estimator="linear",
        learning_rate=0.003,
        steps=10000,
        batch_size=128,
        reg=0.001,
        warm_start_intv=True,
        weight_decay=0.0001,
        verbose=10,
    ):

        # convert x and targets into the same format
        x, targets, n_envs, self.n_vars = KDSMixin._format_input_data(x, targets)

        # set up device sharding
        device_count = jax.device_count()
        devices = jax.devices()
        mesh = mesh_utils.create_device_mesh((device_count,), devices)
        sharding = PositionalSharding(mesh)

        # initialize parameters and load to device (replicate across devices)
        key, subk = random.split(key)
        param = self.init_param(subk, self.n_vars)

        ###
        # No interventional param initialized
        # key, subk = random.split(key)
        # intv_param = self.init_intv_param(subk, self.n_vars, n_envs=n_envs, targets=targets,
        #                     x=x if warm_start_intv else None)
        ###
        
        param = jax.device_put(param, sharding.replicate())
        intv_param = jax.device_put(intv_param, sharding.replicate())
        
        # init dataloader
        key, subk = random.split(key)
        train_loader = make_dataloader(seed=subk[0].item(), sharding=sharding, x=x, batch_size=batch_size)

        # init kernel
        if isinstance(bandwidth, list):
            kernel = lambda x, y: sum([partial(rbf_kernel, bandwidth=float(b))(x, y) for b in bandwidth])
        else:
            kernel = partial(rbf_kernel, bandwidth=float(bandwidth))

        # init KDS loss
        loss_fun = kds_loss(self.f, self.sigma, kernel, estimator=estimator)

        def objective_fun(param_tup, _, batch_):
            param_, intv_param_ = param_tup

            # select interventional parameters of the batch
            # by taking dot-product with environment one-hot indicator vector
            select = lambda leaf: jnp.einsum("e,e...", batch_.env_indicator, leaf)
            intv_param_ = tree_map(select, intv_param_)
            intv_param_.targets = tree_map(select, intv_param_.targets)

            # compute mean KDS loss over environments
            loss = loss_fun(batch_.x, param_, intv_param_)
            assert loss.ndim == 0

            # compute any regularizers
            # scale by variables to be less-tuning sensitive w.r.t. to dimension
            reg_penalty = reg * self.regularize_sparsity(param_) / self.n_vars
            assert reg_penalty.ndim == 0

            # return loss, aux info dict
            l = loss + reg_penalty
            return l, dict(kds_loss=loss)

        value_and_grad =  jax.value_and_grad(objective_fun, 0, has_aux=True)

        # init optimizer and update step
        optimizer = optax.chain(optax.adamw(learning_rate, weight_decay = weight_decay))
        opt_state = optimizer.init((param, intv_param))


        @jax.jit
        def update_step(key_, batch_, param_, intv_param_, opt_state_):
            """
            A single update step of the optimizer
            """
            # compute gradient of objective
            (l, l_aux), (dparam, dintv_param) = value_and_grad((param_, intv_param_), key_, batch_)

            # apply any gradient masks specified by the model
            dparam = dparam.masked(grad=True)
            dintv_param = dintv_param.masked(grad=True)
            
            # compute parameter update given gradient and apply masks there too
            (param_update, intv_param_update), opt_state_ = optimizer.update((dparam, dintv_param),
                                                                             opt_state_,
                                                                             (param_, intv_param_))
            param_update = param_update.masked(grad=True)
            intv_param_update = intv_param_update.masked(grad=True)

            ###
            # No learning interventional params
            intv_param_update["shift"] *= 0
            ###

            # update step
            param_, intv_param_ = optax.apply_updates((param_, intv_param_),
                                                      (param_update, intv_param_update))

            # logging
            grad_norm = linear_algebra.global_norm(dparam)
            intv_grad_norm = linear_algebra.global_norm(dintv_param)
            nan_occurred_param = tree_isnan(dparam) | tree_isnan(param)
            nan_occurred_intv_param = tree_isnan(dintv_param) | tree_isnan(intv_param)
            aux = dict(loss=l,
                       **l_aux,
                       grad_norm=grad_norm,
                       intv_grad_norm=intv_grad_norm,
                       nan_occurred_param=nan_occurred_param,
                       nan_occurred_intv_param=nan_occurred_intv_param)

            return (param_, intv_param_, opt_state_), aux


        # optimization loop
        logs = defaultdict(float)
        t_loop = time.time()
        log_every = math.ceil(steps / verbose if verbose else 0)

        for t in range(steps + 1):

            # sample data batch
            batch = next(train_loader)

            # update step
            key, subk = random.split(key)
            (param, intv_param, opt_state), logs_t = \
                update_step(subk, batch, param, intv_param, opt_state)

            # update average of training metrics
            logs = update_ave(logs, logs_t)

            if verbose and ((not t % log_every and t != 0) or t == steps):
                t_elapsed = time.time() - t_loop
                t_loop = time.time()
                ave_logs = retrieve_ave(logs)
                logs = defaultdict(float)
                print_str = f"step: {t: >5d} " \
                            f"kds: {ave_logs['loss']: >12.6f}  | " \
                            f"min remain: {(steps - t) * t_elapsed / log_every / 60.0: >4.1f}  " \
                            f"sec/step: {t_elapsed / log_every: >5.3f}"
                print(print_str, flush=True)

        if targets is not None and intv_param.targets is not None:
            assert jnp.array_equal(jnp.array(targets), intv_param.targets)

        # save parameters
        self.param = param
        self.intv_param = intv_param

        return self



