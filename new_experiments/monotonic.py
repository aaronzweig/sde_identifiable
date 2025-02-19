import functools

import jax
from jax import vmap, random
import jax.numpy as jnp

class MonotonicMLP():
    def __init__(self, input_dim, hidden_dim = 10, act = jax.nn.sigmoid):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.act = act
        self.shape = {
            "A": jnp.zeros((input_dim, hidden_dim)),
            "B": jnp.zeros((input_dim, hidden_dim)),
            "c": jnp.zeros((hidden_dim)),
            "d": jnp.zeros((input_dim))
        }

    def forward(self, x, param):
        A, B, c, d = param["A"], param["B"], param["c"][None], param["d"]
        
        z = x[:,None]
        
        z = B * z + c
        z = self.act(z)
        z = A * z
        # z = jnp.abs(B) * z + c
        # z = self.act(z)
        # z = jnp.abs(A) * z
        z = jnp.sum(z, axis=-1) + d
        z = z + 0.01*self.act(x)
        assert z.shape == x.shape
        return z


    
    # def vmap_forward(x, params):

    #     batched_activation = jax.vmap(self.forward, in_axes=(0, None))  # Batched over first axis
    #     unbatched_activation = jax.vmap(self.forward, in_axes=(None, None))  # Effectively no vmap
        
    #     return jax.lax.cond(
    #         x.ndim > 1,  # Assume batch dimension means ndim > 1
    #         lambda _: batched_activation(x, params),
    #         lambda _: unbatched_activation(x, params),
    #         operand=None
    #     )
    
    # def vmap_forward(self, x, param):
    #     jax.debug.print(str(x.ndim))
    #     return jax.lax.cond(
    #         x.ndim > 1,
    #         vmap(self.forward, in_axes=[0,None])(x, param),
    #         vmap(self.forward, in_axes=[None,None])(x, param),
    #     )
        
    def vmap_forward(self, x, param):
        if x.ndim == 1:
            return self.forward(x, param)
        else:
            return vmap(self.forward, in_axes=[0,None])(x, param)
            