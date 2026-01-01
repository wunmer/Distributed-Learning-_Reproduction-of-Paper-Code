#研究误差与分布式机器数量的关系（控制每台机器样本量n=100）
# 2.mse(mse标准差)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def log_likelihood_gaussian(params, data):
    mu, sigma2 = params
    n = len(data)
    if sigma2 <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, loc=mu, scale=np.sqrt(sigma2)))

def centralized_mle(data):
    result = minimize(log_likelihood_gaussian, x0=[np.mean(data), np.var(data)], 
                     args=(data,), method='L-BFGS-B', 
                     bounds=[(None, None), (1e-6, None)])
    return result.x

def simulate_gaussian_experiment(n, k, K=50):
    # n: 每台机器的样本量 (固定)
    # k: 机器数量
    N = n * k  # 总样本量 = 每台机器样本量 × 机器数量
    
    mse_simple = []
    mse_resampled = []
    mse_onestep = []
    mse_central = []
    
    for _ in range(K):
        # True parameters
        mu_true = np.random.uniform(-2, 2)
        sigma2_true = np.random.uniform(0.25, 9)
        
        # Generate full dataset
        data_full = np.random.normal(mu_true, np.sqrt(sigma2_true), N)
        
        # Split data into k machines, each with n samples
        data_splits = np.array_split(data_full, k)
        
        # Local MLEs
        local_estimates = []
        for i in range(k):
            try:
                est = centralized_mle(data_splits[i])
                local_estimates.append(est)
            except:
                local_estimates.append([mu_true, sigma2_true])  # fallback
        
        local_estimates = np.array(local_estimates)
        
        # Simple averaging estimator
        theta_simple = np.mean(local_estimates, axis=0)
        mse_simple.append(np.sum((theta_simple - [mu_true, sigma2_true])**2))
        
        # Centralized MLE
        theta_central = centralized_mle(data_full)
        mse_central.append(np.sum((theta_central - [mu_true, sigma2_true])**2))
        
        # One-step estimator
        def grad_hess_gaussian(theta, data):
            mu, sigma2 = theta
            n_data = len(data)
            grad_mu = np.sum(data - mu) / sigma2
            grad_sigma2 = -n_data/(2*sigma2) + np.sum((data - mu)**2) / (2*sigma2**2)
            grad = np.array([grad_mu, grad_sigma2])
            
            hess_mu_mu = -n_data / sigma2
            hess_mu_sigma2 = -np.sum(data - mu) / sigma2**2
            hess_sigma2_sigma2 = n_data/(2*sigma2**2) - np.sum((data - mu)**2) / sigma2**3
            hess = np.array([[hess_mu_mu, hess_mu_sigma2], 
                             [hess_mu_sigma2, hess_sigma2_sigma2]])
            return grad, hess
        
        grad, hess = grad_hess_gaussian(theta_simple, data_full)
        try:
            theta_onestep = theta_simple - np.linalg.inv(hess) @ grad
            mse_onestep.append(np.sum((theta_onestep - [mu_true, sigma2_true])**2))
        except:
            mse_onestep.append(mse_simple[-1])  # fallback to simple averaging
        
        # Resampled averaging estimator
        s = 0.1
        theta1_simple = []
        for i in range(k):
            resampled_data = np.random.choice(data_splits[i], size=int(s * n), replace=False)
            try:
                est = centralized_mle(resampled_data)
                theta1_simple.append(est)
            except:
                theta1_simple.append([mu_true, sigma2_true])
        theta1_simple = np.mean(theta1_simple, axis=0)
        theta_resampled = (theta_simple - s * theta1_simple) / (1 - s)
        mse_resampled.append(np.sum((theta_resampled - [mu_true, sigma2_true])**2))
    
    return {
        'simple_avg': (np.mean(mse_simple), np.std(mse_simple)),
        'resampled_avg': (np.mean(mse_resampled), np.std(mse_resampled)),
        'onestep': (np.mean(mse_onestep), np.std(mse_onestep)),
        'centralized': (np.mean(mse_central), np.std(mse_central))
    }

# Run experiment with fixed sample size per machine (n=100)
results = []
k_values = [8, 16, 32, 64, 128, 256, 512]
n_per_machine = 100  # 固定每台机器的样本量

print("Running Gaussian Distribution experiments with fixed sample size per machine...")
print(f"Sample size per machine: {n_per_machine}")
for k in k_values:
    print(f"Running k={k}, n={n_per_machine} per machine, N={k*n_per_machine} total...")
    res = simulate_gaussian_experiment(n_per_machine, k, K=30)  # Reduced K for speed
    res['k'] = k
    res['n_per_machine'] = n_per_machine
    res['N'] = k * n_per_machine
    results.append(res)

# Print results in table format like the paper
print("\n" + "="*90)
print("Gaussian Distribution with Unknown Mean and Variance")
print(f"Fixed sample size per machine: {n_per_machine}")
print("="*90)
print("no. of\tno. of\tper machine\tsimple avg\t\tresampled avg\t\tone-step\t\tcentralized")
print("machines\tsamples\tsample size")
print("-"*90)

for res in results:
    k, N, n = res['k'], res['N'], res['n_per_machine']
    simple_mean, simple_std = res['simple_avg']
    resampled_mean, resampled_std = res['resampled_avg']
    onestep_mean, onestep_std = res['onestep']
    central_mean, central_std = res['centralized']
    
    print(f"{k}\t{N}\t{n}\t\t{simple_mean:.6f}\t\t{resampled_mean:.6f}\t\t{onestep_mean:.6f}\t\t{central_mean:.6f}")
    print(f"\t\t\t\t({simple_std:.6f})\t\t({resampled_std:.6f})\t\t({onestep_std:.6f})\t\t({central_std:.6f})")
    print()

# Also print in the compact format used in paper tables
print("\n" + "="*90)
print("Compact Format (like paper Table 5)")
print(f"Fixed sample size per machine: {n_per_machine}")
print("="*90)
print("no. of\tno. of\tper machine\tsimple avg\t\tresampled avg\t\tone-step\t\tcentralized")
print("machines\tsamples\tsample size")
print("-"*90)

for res in results:
    k, N, n = res['k'], res['N'], res['n_per_machine']
    simple_mean, simple_std = res['simple_avg']
    resampled_mean, resampled_std = res['resampled_avg']
    onestep_mean, onestep_std = res['onestep']
    central_mean, central_std = res['centralized']
    
    # Format like paper: mean (std)
    simple_str = f"{simple_mean:.6f} ({simple_std:.6f})"
    resampled_str = f"{resampled_mean:.6f} ({resampled_std:.6f})"
    onestep_str = f"{onestep_mean:.6f} ({onestep_std:.6f})"
    central_str = f"{central_mean:.6f} ({central_std:.6f})"
    
    print(f"{k}\t{N}\t{n}\t\t{simple_str}\t{resampled_str}\t{onestep_str}\t{central_str}")