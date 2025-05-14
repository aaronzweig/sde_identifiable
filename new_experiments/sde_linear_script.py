import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from scipy.linalg import solve_continuous_lyapunov, svd


import matplotlib.pyplot as plt
import numpy as np

class LinearSDE(nn.Module):

    def __init__(self, n, r, decay = None):
        super(LinearSDE, self).__init__()

        self.A = torch.nn.Parameter(torch.zeros(n, r))
        self.B = torch.nn.Parameter(torch.zeros(r, n))
        with torch.no_grad():
            nn.init.xavier_uniform_(self.A)
            nn.init.xavier_uniform_(self.B)

        if decay is not None:
            self.D = torch.diag(torch.from_numpy(decay)).float()
        else:
            self.D = torch.nn.Parameter(torch.zeros(n,))
            with torch.no_grad():
                    nn.init.ones_(self.D)

        assert(self.D.dim() == 1)

    def forward(self):
        return self.A @ self.B - torch.diag(self.D)

    def get_decay(self):
        return torch.diag(self.D)

def train(n, r, C, m, w, lr = 0.001, iterations = 500, decay = None):

    model = LinearSDE(n, r, decay)
    optimizer = optim.Adam(model.parameters(), lr = lr)

    cov = torch.from_numpy(w).float()
    mean = torch.from_numpy(m).float()
    C_int = torch.from_numpy(C).float()

    losses = []

    with torch.autograd.set_detect_anomaly(True):
        for _ in range(iterations):
            optimizer.zero_grad()
    
            L_hat = model()
            m_hat = -torch.linalg.inv(L_hat) @ C_int
            
            loss = 0
            loss += torch.norm(mean - m_hat)
            loss += torch.norm(L_hat @ cov + cov @ L_hat.T + torch.eye(n))
            
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    return model, loss.item(), losses

def build_linear_sde(n, r, gamma, k):
    D = np.identity(n) + np.diag(np.random.uniform(size=(n,)))
    
    A = np.random.normal(size = (n,r))
    A *= gamma / np.linalg.norm(A, ord = 2)
    B = np.random.normal(size = (r,n))
    B *= gamma / np.linalg.norm(B, ord = 2)
    
    L = A @ B - D
    
    w = solve_continuous_lyapunov(L, -np.identity(n))

    C = np.random.normal(size = (n,k))
    m = -np.linalg.inv(L) @ C

    return L, C, m, w, D

def get_drift_error(n, r, gamma, k, fix_decay=False, redraws = 1):
    print(f"redraws now is {redraws}")
    L, C, m, w, D = build_linear_sde(n, r, gamma, k)

    L_hats = []
    model_losses = []
    for _ in range(redraws):
        model, loss, _ = train(n, r, C, m, w, lr = 0.005, iterations = 3000, decay = D if fix_decay else None)
        L_hat = model().detach().numpy()

        model_losses.append(loss)
        L_hats.append(L_hat)

    min_index = model_losses.index(min(model_losses))
    L_hat = L_hats[min_index]
        

    return np.linalg.norm(L_hat - L)/np.linalg.norm(L)






import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--redraws', type=int, default=30)

    args = parser.parse_args()

    n = args.n
    np.random.seed(42)
    gamma = 0.9
    samples = 5
    rs = [5, 10]
    # fix_decay = True
    fix_decay = False
    redraws = args.redraws

    errors = []
    for oversample in [False, True]:
        error_list = []
        for r in rs:
            print(r)
            if not oversample:
                k = r - 2
            else:
                if fix_decay:
                    k = r
                else:
                    k = int(r * np.log(n))
    
            runs = []
            for _ in range(samples):
                # err = get_drift_error(n, r, gamma, k, fix_decay, redraws)
                err = get_drift_error(n, r, gamma, k, fix_decay, redraws if fix_decay or not oversample else 5)
                runs.append(err)
            runs = np.array(runs)
            mean, std = np.mean(runs), np.std(runs)
            error_list.append((mean, std))
        errors.append(error_list)
    
    print(errors)
    
    # Example data
    categories = rs
    mean0, std0 = zip(*errors[0])
    mean1, std1 = zip(*errors[1])
    
    # Bar width
    bar_width = 0.4
    x = np.arange(len(categories))  # Positions for the groups
    
    # Create the bar plot
    plt.bar(x - bar_width/2, mean0, yerr = std0, width=bar_width, label='k=r-2', color='blue')
    label = 'k=r' if fix_decay else 'k=r*log(n)'
    plt.bar(x + bar_width/2, mean1, yerr = std1, width=bar_width, label=label, color='orange')
    
    # Add labels, title, and legend
    plt.xlabel('True rank')
    plt.ylabel('Normalized Frobenius Error')
    plt.title('Linear SDE drift recovery')
    plt.xticks(x, categories)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    # plt.show()
    # plt.savefig("linear_sde_recovery_theory_" + str(n) +".png")
    plt.savefig("linear_sde_recovery_theory_" + str(n) + "_" + str(fix_decay) + ".png")


if __name__ == "__main__":
    main()







