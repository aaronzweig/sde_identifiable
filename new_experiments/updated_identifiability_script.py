import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from scipy.linalg import solve_continuous_lyapunov, svd
import scipy
import torch
import torchsde
import torch.nn as nn
import numpy as np

import math

import matplotlib.pyplot as plt
import numpy as np

from torch.func import jvp, vmap, jacrev
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sample_stiefel(n, m):
    assert m <= n
    X = np.random.normal(size = (n, m))
    Z = X @ np.linalg.inv(scipy.linalg.sqrtm(X.T @ X))
    return np.real(Z)

class ActivationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ActivationMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.A = torch.nn.Parameter(torch.zeros(1,input_dim,hidden_dim))
        self.B = torch.nn.Parameter(torch.zeros(1,input_dim,hidden_dim))
        self.c = torch.nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.d = torch.nn.Parameter(torch.zeros(1,input_dim))
        self.act = nn.Sigmoid()

    def forward(self, x):
        A, B, c, d = self.A, self.B, self.c, self.d
        
        z = x.unsqueeze(-1)

        z = B * z + c
        z = self.act(z)
        z = A * z
        z = torch.sum(z, dim=-1) + d
        
        z = z + 0.01*self.act(x)
        
        assert z.shape == x.shape
        return z

class FullSDE(nn.Module):

    def __init__(self, n, r, gamma = 0.98, act = None, ep = 1e-3, hidden_dim=10):
        super(FullSDE, self).__init__()

        self.act = act
        if act is None:
            self.act = ActivationMLP(r, hidden_dim)

        self.n = n
        self.r = r
        self.ep = ep
        self.gamma = gamma

        self.A = torch.nn.Parameter(torch.zeros(n, r))
        self.B = torch.nn.Parameter(torch.zeros(r, n))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.B)
            self.A /= torch.linalg.norm(self.A, ord = 2)
            self.B /= torch.linalg.norm(self.B, ord = 2)
            ############################################################
            if act is not None: #try sparse sampling
                # self.A = torch.nn.Parameter(torch.eye(n, r)[torch.randperm(n)])
                # self.B = torch.nn.Parameter(torch.eye(r, n)[:,torch.randperm(n)])
                A = torch.from_numpy(sample_stiefel(n,r)).float()
                B = torch.from_numpy(sample_stiefel(n,r)).float().T
                self.A = torch.nn.Parameter(A)
                self.B = torch.nn.Parameter(B)
            ############################################################

    def drift(self, x):
        # x: (batch, n)
        return self.gamma * self.act(x @ self.B.T) @ self.A.T - x

    def jacobian(self, x):
        def F(x):
            x = x.unsqueeze(0)
            return self.drift(x).squeeze(0)
        return vmap(jacrev(F))(x)

class SDERunner(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, model, c):
        super().__init__()
        self.model = model
        self.c = c

        self.dim = model.n

    def f(self, t, y):
        return self.model.drift(y) + self.c.unsqueeze(0)

    def g(self, t, y):
        return np.sqrt(self.model.ep) * torch.ones_like(y)

@torch.no_grad()
def sample_stationary_chunked(
    sde,
    n_samples,
    burnin=100,
    thinning=300,
    rollouts_shape=(4096,),
    dt=1e-3,
    device="cuda",
    sigma=0.0,
):

    n_rollouts = math.prod(rollouts_shape)
    n_samples_per_rollout = math.ceil(n_samples / n_rollouts) + burnin

    sde.to(device)

    x_star = torch.zeros(1, sde.dim, device=device)
    for j in range(30):
        x_star = sde.f(None, x_star) + x_star

    
    y = x_star + sigma * torch.randn(*rollouts_shape, sde.dim, device=device)
    
    samples_kept = []

    for _ in range(n_samples_per_rollout):
        # integrate for `thinning` steps, only keep the last
        ts = torch.linspace(0, thinning * dt, thinning + 1, device=device)
        bm = None
        with torch.no_grad():
            ys = torchsde.sdeint(sde, y, ts, bm=bm, method="euler", dt=dt)

        y = ys[-1]  # final state of this chunk
        samples_kept.append(y)

    # [n_samples_per_rollout, *rollouts_shape, d]
    traj = torch.stack(samples_kept, dim=0)

    traj = traj[burnin:]

    samples = traj.reshape(-1, sde.dim)

    perm = torch.randperm(samples.shape[0], device=device)
    samples = samples[perm[:n_samples]]

    return samples

def generate_data(model, C, n_samples):
    samples_list = []
    means = []
    covs = []
    for i in range(C.shape[0]):
        # print(f"condition {i}")
        c = C[i]
        sde = SDERunner(model, c)
        
        samples = sample_stationary_chunked(sde, n_samples)
        mean = torch.mean(samples, dim=0)
        cov = torch.cov(samples.T)

        samples_list.append(samples)
        means.append(mean)
        covs.append(cov)
    return torch.stack(samples_list, dim = 0), torch.stack(means, dim=0), torch.stack(covs, dim = 0)

def train(model, C, means, covs, lr = 0.001, iterations = 500):

    optimizer = optim.Adam(model.parameters(), lr = lr)
    ep = model.ep
    n = model.n
    
    losses = []

    for _ in range(iterations):
        optimizer.zero_grad()

        approx_stationary = model.drift(means) + C
        J = model.jacobian(means)
                
        loss = 0
        loss += torch.norm(approx_stationary)
        loss += torch.norm(J @ (covs / ep) + (covs / ep) @ J.transpose(1,2) + torch.eye(n).to('cuda'))
        
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, loss.item(), losses

import itertools
import numpy as np

def permuted_error(A, A_star):
    perms = list(itertools.permutations(range(A.shape[1])))

    A /= np.linalg.norm(A, axis = 0, keepdims = True)
    A *= np.sign(np.sum(A, axis = 0, keepdims = True))
    A_star /= np.linalg.norm(A_star, axis = 0, keepdims = True)
    A_star *= np.sign(np.sum(A_star, axis = 0, keepdims = True))
    
    errors = []
    for perm in perms:
        A_perm = A_star[:, perm]
        err = np.linalg.norm(A - A_perm)
        errors.append(err)
    
    return min(errors) / np.linalg.norm(A_star)




import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_runs', type=int, default=101)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1000000)
    parser.add_argument("--verbose", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--ep", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=10)

    args = parser.parse_args()

    #TODO: free samples from CUDA memory?  Take running averages to save memory?
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n = 8
    gamma = 0.995
    n_samples = args.n_samples
    r = 2
    k = args.k
    ep = args.ep
    hidden_dim = args.hidden_dim
    learned_act = True

    verbose = args.verbose
    
    n_runs = args.n_runs
    lr = 0.0005
    iterations = 30000

    true_model = FullSDE(n=n, r=r, gamma=gamma, act=nn.Sigmoid(), ep=ep)
    true_model.to('cuda')
    
    A_true = true_model.A.detach().cpu().numpy()
    B_true = true_model.B.detach().cpu().numpy().T
    
    C = 0.3 * torch.randn(k, n).to('cuda')
    samples, means, covs = generate_data(true_model, C, n_samples)
    del samples

    if verbose:
        approx_stationary = true_model.drift(means) + C
        J = true_model.jacobian(means)
        
        loss = 0
        loss += torch.norm(approx_stationary)
        loss += torch.norm(J @ (covs / ep) + (covs / ep) @ J.transpose(1,2) + torch.eye(n).to('cuda'))
        print(loss)

    errors = []
    for i in tqdm(range(n_runs)):
        if verbose:
            print("run", i)
    
        if i == 0:
            # print("first run is just against true model")
            model = FullSDE(n=n, r=r, gamma=gamma, act=nn.Sigmoid(), ep=ep)
            model.load_state_dict(true_model.state_dict())
        else:
            act = None if learned_act else nn.Sigmoid()
            model = FullSDE(n=n, r=r, gamma=gamma, act=act, ep=ep, hidden_dim=hidden_dim)
        model.to('cuda')
        
        
        model, loss, losses = train(model, C, means, covs, lr = lr, iterations = iterations)    
        A = model.A.detach().cpu().numpy()
        B = model.B.detach().cpu().numpy().T
        
        A_error = permuted_error(A, A_true)
        B_error = permuted_error(B, B_true)

        if verbose:
            print(loss, A_error, B_error)
        if i == 0:
            true_error = (loss, A_error, B_error)
        if i > 0:
            errors.append((loss, A_error, B_error))

    print(args.n_runs, args.seed, args.k)
    print("true_model")
    print(true_error)
    print("best_trained_model")
    print(min(errors, key=lambda x: x[0])) #min outside of best true model


if __name__ == "__main__":
    main()







