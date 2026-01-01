import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DistributedLogisticRegression:
    """分布式逻辑回归实验类"""
    
    def __init__(self, seed=42):
        """初始化"""
        np.random.seed(seed)
        self.results = {}
        
    def generate_data(self, N, d, n):
        """
        生成逻辑回归数据
        
        参数:
        N: 总样本量
        d: 参数维度
        n: 每个机器的样本量
        """
        print(f"生成数据: N={N}, d={d}, n={n}")
        
        # 真实参数从[0,1]^d均匀采样
        theta_true = np.random.uniform(0, 1, d)
        
        # 计算机器数量
        k = N // n
        if k * n != N:
            print(f"警告: N={N}不能被n={n}整除，调整N为{k*n}")
            N = k * n
        
        # 生成协变量 X ~ N(0, I_d)
        X = np.random.randn(N, d)
        
        # 生成响应变量 Y ~ Bernoulli(σ(Xθ))
        probs = expit(X @ theta_true)
        Y = np.random.binomial(1, probs)
        
        # 将数据分配到k个机器
        X_split = np.array_split(X, k)
        Y_split = np.array_split(Y, k)
        
        print(f"数据已分配到{k}个机器，每个机器{n}个样本")
        return theta_true, X_split, Y_split, k
    
    def logistic_loss(self, theta, X, Y, reg_param=0):
        """逻辑回归损失函数（负对数似然）"""
        z = X @ theta
        log_prob = np.log(1 + np.exp(z))
        loss = -np.sum(Y * z - log_prob) / len(Y)
        
        if reg_param > 0:
            loss += reg_param * np.sum(theta**2)  # L2正则化
        
        return loss
    
    def logistic_gradient(self, theta, X, Y, reg_param=0):
        """逻辑回归梯度"""
        z = X @ theta
        prob = expit(z)
        grad = X.T @ (prob - Y) / len(Y)
        
        if reg_param > 0:
            grad += 2 * reg_param * theta
        
        return grad
    
    def logistic_hessian(self, theta, X, reg_param=0):
        """逻辑回归Hessian矩阵"""
        z = X @ theta
        prob = expit(z)
        W = np.diag(prob * (1 - prob))
        hessian = X.T @ W @ X / len(X)
        
        if reg_param > 0:
            hessian += 2 * reg_param * np.eye(len(theta))
        
        return hessian
    
    def global_m_estimator(self, X, Y, theta_init=None):
        """全局M估计器（使用所有数据）"""
        print("  计算全局M估计器...")
        
        # 合并所有数据
        X_all = np.vstack(X)
        Y_all = np.hstack(Y)
        
        if theta_init is None:
            theta_init = np.zeros(X_all.shape[1])
        
        # 最小化负对数似然
        result = minimize(
            fun=self.logistic_loss,
            x0=theta_init,
            args=(X_all, Y_all),
            method='L-BFGS-B',
            jac=self.logistic_gradient,
            options={'maxiter': 1000, 'disp': False}
        )
        
        return result.x
    
    def local_estimator(self, X, Y, machine_idx=0):
        """本地估计器（仅使用一个机器的数据）"""
        print(f"  计算本地估计器（机器{machine_idx}）...")
        
        theta_init = np.zeros(X[machine_idx].shape[1])
        
        result = minimize(
            fun=self.logistic_loss,
            x0=theta_init,
            args=(X[machine_idx], Y[machine_idx]),
            method='L-BFGS-B',
            jac=self.logistic_gradient,
            options={'maxiter': 1000, 'disp': False}
        )
        
        return result.x
    
    def average_estimator(self, X, Y):
        """平均估计器（各本地估计器的平均）"""
        print("  计算平均估计器...")
        
        d = X[0].shape[1]
        k = len(X)
        local_estimates = []
        
        for j in range(k):
            theta_j = self.local_estimator(X, Y, j)
            local_estimates.append(theta_j)
            print(f"    机器{j}的估计完成")
        
        avg_theta = np.mean(local_estimates, axis=0)
        return avg_theta, local_estimates
    
    def csl_estimator(self, X, Y, theta_bar, method='one_step'):
        """
        CSL估计器（通信高效的替代似然）
        
        参数:
        X, Y: 分布式数据
        theta_bar: 初始估计器（如平均估计器）
        method: 'one_step'或'iterative'
        """
        print(f"  计算{method} CSL估计器...")
        
        k = len(X)
        d = X[0].shape[1]
        
        # 步骤1: 计算局部梯度
        local_gradients = []
        for j in range(k):
            grad_j = self.logistic_gradient(theta_bar, X[j], Y[j])
            local_gradients.append(grad_j)
        
        # 步骤2: 计算全局梯度（平均）
        global_gradient = np.mean(local_gradients, axis=0)
        
        # 步骤3: 获取第一个机器的梯度
        grad_M1 = local_gradients[0]
        
        # 步骤4: 构建替代损失函数
        def surrogate_loss(theta):
            """替代损失函数 L_tilde(θ) = L1(θ) - <θ, ∇L1(θ_bar) - ∇L_N(θ_bar)>"""
            loss_M1 = self.logistic_loss(theta, X[0], Y[0])
            linear_term = theta @ (grad_M1 - global_gradient)
            return loss_M1 - linear_term
        
        def surrogate_gradient(theta):
            """替代损失函数的梯度 ∇L_tilde(θ) = ∇L1(θ) - [∇L1(θ_bar) - ∇L_N(θ_bar)]"""
            grad_theta = self.logistic_gradient(theta, X[0], Y[0])
            return grad_theta - (grad_M1 - global_gradient)
        
        # 步骤5: 最小化替代损失函数
        theta_init = theta_bar.copy()
        
        result = minimize(
            fun=surrogate_loss,
            x0=theta_init,
            method='L-BFGS-B',
            jac=surrogate_gradient,
            options={'maxiter': 1000, 'disp': False}
        )
        
        one_step_csl = result.x
        
        if method == 'one_step':
            return one_step_csl
        
        # 迭代版本（这里简化为再执行一次CSL）
        elif method == 'iterative':
            print("  执行迭代CSL（第二步）...")
            # 使用一步CSL的结果作为新的初始值
            return self.csl_estimator(X, Y, one_step_csl, method='one_step')
    
    def run_experiment_fixed_N(self, N, d, n_values):
        """
        实验1: 固定总样本量N，变化本地样本量n
        
        参数:
        N: 总样本量
        d: 参数维度
        n_values: 本地样本量列表
        """
        print("="*60)
        print("实验1: 固定总样本量N，变化本地样本量n")
        print(f"N = {N}, d = {d}")
        print("="*60)
        
        errors = {
            'global': [],
            'local': [],
            'average': [],
            'csl_one_step': [],
            'csl_iterative': []
        }
        
        n_list = []
        
        for n in n_values:
            print(f"\n运行实验: n = {n}")
            print("-"*40)
            
            # 生成数据
            theta_true, X_split, Y_split, k = self.generate_data(N, d, n)
            n_list.append(n)
            
            # 1. 全局M估计器（基准）
            theta_global = self.global_m_estimator(X_split, Y_split)
            error_global = np.linalg.norm(theta_global - theta_true)
            errors['global'].append(error_global)
            print(f"  全局估计器误差: {error_global:.4f}")
            
            # 2. 本地估计器（仅使用第一个机器）
            theta_local = self.local_estimator(X_split, Y_split, 0)
            error_local = np.linalg.norm(theta_local - theta_true)
            errors['local'].append(error_local)
            print(f"  本地估计器误差: {error_local:.4f}")
            
            # 3. 平均估计器
            theta_avg, _ = self.average_estimator(X_split, Y_split)
            error_avg = np.linalg.norm(theta_avg - theta_true)
            errors['average'].append(error_avg)
            print(f"  平均估计器误差: {error_avg:.4f}")
            
            # 4. 一步CSL估计器（使用平均估计器作为初始值）
            theta_csl_one = self.csl_estimator(X_split, Y_split, theta_avg, 'one_step')
            error_csl_one = np.linalg.norm(theta_csl_one - theta_true)
            errors['csl_one_step'].append(error_csl_one)
            print(f"  一步CSL估计器误差: {error_csl_one:.4f}")
            
            # 5. 迭代CSL估计器（两步）
            theta_csl_iter = self.csl_estimator(X_split, Y_split, theta_avg, 'iterative')
            error_csl_iter = np.linalg.norm(theta_csl_iter - theta_true)
            errors['csl_iterative'].append(error_csl_iter)
            print(f"  迭代CSL估计器误差: {error_csl_iter:.4f}")
            
            print(f"完成n={n}的实验")
        
        self.results['fixed_N'] = {
            'n_values': n_list,
            'errors': errors,
            'N': N,
            'd': d
        }
        
        return errors, n_list
    
    def run_experiment_fixed_n(self, n, d, k_values):
        """
        实验2: 固定本地样本量n，变化机器数量k
        
        参数:
        n: 本地样本量
        d: 参数维度
        k_values: 机器数量列表
        """
        print("="*60)
        print("实验2: 固定本地样本量n，变化机器数量k")
        print(f"n = {n}, d = {d}")
        print("="*60)
        
        errors = {
            'global': [],
            'local': [],
            'average': [],
            'csl_one_step': [],
            'csl_iterative': []
        }
        
        k_list = []
        
        for k in k_values:
            print(f"\n运行实验: k = {k}")
            print("-"*40)
            
            # 总样本量 N = n * k
            N = n * k
            k_list.append(k)
            
            # 生成数据
            theta_true, X_split, Y_split, _ = self.generate_data(N, d, n)
            
            # 1. 全局M估计器
            theta_global = self.global_m_estimator(X_split, Y_split)
            error_global = np.linalg.norm(theta_global - theta_true)
            errors['global'].append(error_global)
            print(f"  全局估计器误差: {error_global:.4f}")
            
            # 2. 本地估计器
            theta_local = self.local_estimator(X_split, Y_split, 0)
            error_local = np.linalg.norm(theta_local - theta_true)
            errors['local'].append(error_local)
            print(f"  本地估计器误差: {error_local:.4f}")
            
            # 3. 平均估计器
            theta_avg, _ = self.average_estimator(X_split, Y_split)
            error_avg = np.linalg.norm(theta_avg - theta_true)
            errors['average'].append(error_avg)
            print(f"  平均估计器误差: {error_avg:.4f}")
            
            # 4. 一步CSL估计器
            theta_csl_one = self.csl_estimator(X_split, Y_split, theta_avg, 'one_step')
            error_csl_one = np.linalg.norm(theta_csl_one - theta_true)
            errors['csl_one_step'].append(error_csl_one)
            print(f"  一步CSL估计器误差: {error_csl_one:.4f}")
            
            # 5. 迭代CSL估计器
            theta_csl_iter = self.csl_estimator(X_split, Y_split, theta_avg, 'iterative')
            error_csl_iter = np.linalg.norm(theta_csl_iter - theta_true)
            errors['csl_iterative'].append(error_csl_iter)
            print(f"  迭代CSL估计器误差: {error_csl_iter:.4f}")
            
            print(f"完成k={k}的实验")
        
        self.results['fixed_n'] = {
            'k_values': k_list,
            'errors': errors,
            'n': n,
            'd': d
        }
        
        return errors, k_list
    
    def plot_results_fixed_N(self):
        """绘制固定N的实验结果"""
        if 'fixed_N' not in self.results:
            print("未找到固定N的实验结果")
            return
        
        data = self.results['fixed_N']
        n_values = data['n_values']
        errors = data['errors']
        N = data['N']
        d = data['d']
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"固定总样本量N={N}（线性坐标）",
                f"固定总样本量N={N}（对数坐标）"
            )
        )
        
        # 颜色设置
        colors = {
            'global': 'rgba(31, 119, 180, 1)',
            'local': 'rgba(255, 127, 14, 1)',
            'average': 'rgba(44, 160, 44, 1)',
            'csl_one_step': 'rgba(214, 39, 40, 1)',
            'csl_iterative': 'rgba(148, 103, 189, 1)'
        }
        
        # 线性坐标图
        for method, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=n_values,
                    y=errors[method],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 对数坐标图
        for method, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=n_values,
                    y=errors[method],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 更新布局
        fig.update_xaxes(
            title_text="本地样本量n",
            type='linear',
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="本地样本量n",
            type='log',
            row=1, col=2
        )
        
        fig.update_yaxes(
            title_text="L2误差",
            type='linear',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="L2误差",
            type='log',
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"分布式逻辑回归M估计（d={d}）",
            height=500,
            width=1000,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 添加注释
        fig.add_annotation(
            text=f"总样本量N固定为{N}，k=N/n",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_results_fixed_n(self):
        """绘制固定n的实验结果"""
        if 'fixed_n' not in self.results:
            print("未找到固定n的实验结果")
            return
        
        data = self.results['fixed_n']
        k_values = data['k_values']
        errors = data['errors']
        n = data['n']
        d = data['d']
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"固定本地样本量n={n}（线性坐标）",
                f"固定本地样本量n={n}（对数坐标）"
            )
        )
        
        # 颜色设置
        colors = {
            'global': 'rgba(31, 119, 180, 1)',
            'local': 'rgba(255, 127, 14, 1)',
            'average': 'rgba(44, 160, 44, 1)',
            'csl_one_step': 'rgba(214, 39, 40, 1)',
            'csl_iterative': 'rgba(148, 103, 189, 1)'
        }
        
        # 线性坐标图
        for method, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=errors[method],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 对数坐标图
        for method, color in colors.items():
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=errors[method],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 更新布局
        fig.update_xaxes(
            title_text="机器数量k",
            type='linear',
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="机器数量k",
            type='log',
            row=1, col=2
        )
        
        fig.update_yaxes(
            title_text="L2误差",
            type='linear',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="L2误差",
            type='log',
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"分布式逻辑回归M估计（d={d}）",
            height=500,
            width=1000,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 添加注释
        fig.add_annotation(
            text=f"本地样本量n固定为{n}，N=n×k",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    def create_summary_table(self):
        """创建结果汇总表"""
        print("\n" + "="*60)
        print("实验结果汇总")
        print("="*60)
        
        if 'fixed_N' in self.results:
            print("\n1. 固定总样本量N的实验结果:")
            data = self.results['fixed_N']
            n_values = data['n_values']
            errors = data['errors']
            
            df_fixed_N = pd.DataFrame({
                'n': n_values,
                'k': [data['N'] // n for n in n_values],
                'Global': errors['global'],
                'Local': errors['local'],
                'Average': errors['average'],
                'CSL (1-step)': errors['csl_one_step'],
                'CSL (iterative)': errors['csl_iterative']
            })
            
            print(df_fixed_N.round(4).to_string(index=False))
        
        if 'fixed_n' in self.results:
            print("\n2. 固定本地样本量n的实验结果:")
            data = self.results['fixed_n']
            k_values = data['k_values']
            errors = data['errors']
            
            df_fixed_n = pd.DataFrame({
                'k': k_values,
                'N': [data['n'] * k for k in k_values],
                'Global': errors['global'],
                'Local': errors['local'],
                'Average': errors['average'],
                'CSL (1-step)': errors['csl_one_step'],
                'CSL (iterative)': errors['csl_iterative']
            })
            
            print(df_fixed_n.round(4).to_string(index=False))
        
        return df_fixed_N if 'fixed_N' in self.results else None


def main():
    """主函数"""
    print("开始分布式逻辑回归M估计实验复现")
    print("="*60)
    
    # 创建实验实例
    experiment = DistributedLogisticRegression(seed=42)
    
    # 实验参数设置
    N = 2**19  # 总样本量 ≈ 524,288
    d = 10     # 参数维度
    
    # 实验1: 固定总样本量N，变化本地样本量n
    print("\n执行实验1: 固定总样本量N，变化本地样本量n")
    n_values = [2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]  # 128到16384
    errors1, n_list = experiment.run_experiment_fixed_N(N, d, n_values)
    
    # 实验2: 固定本地样本量n，变化机器数量k
    print("\n执行实验2: 固定本地样本量n，变化机器数量k")
    n_fixed = 2**11  # 固定本地样本量 = 2048
    k_values = [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]  # 16到1024
    errors2, k_list = experiment.run_experiment_fixed_n(n_fixed, d, k_values)
    
    # 绘制结果
    print("\n绘制实验结果...")
    fig1 = experiment.plot_results_fixed_N()
    fig2 = experiment.plot_results_fixed_n()
    
    # 显示图表
    fig1.show()
    fig2.show()
    
    # 创建汇总表
    df_summary = experiment.create_summary_table()
    
    print("\n" + "="*60)
    print("实验复现完成！")
    print("="*60)
    
    # 保存结果
    try:
        fig1.write_html("experiment_fixed_N.html")
        fig2.write_html("experiment_fixed_n.html")
        print("图表已保存为HTML文件")
    except:
        print("注意: 无法保存HTML文件，请检查文件写入权限")
    
    return experiment, fig1, fig2, df_summary


if __name__ == "__main__":
    # 运行实验
    experiment, fig1, fig2, df_summary = main()
    
    # 额外分析：比较CSL与平均估计器的相对改进
    print("\n" + "="*60)
    print("CSL相对于平均估计器的改进分析")
    print("="*60)
    
    if 'fixed_N' in experiment.results:
        errors = experiment.results['fixed_N']['errors']
        n_values = experiment.results['fixed_N']['n_values']
        
        print("\n固定总样本量N时:")
        for i, n in enumerate(n_values):
            avg_error = errors['average'][i]
            csl_error = errors['csl_one_step'][i]
            improvement = (avg_error - csl_error) / avg_error * 100
            print(f"  n={n}: 平均误差={avg_error:.4f}, CSL误差={csl_error:.4f}, 改进={improvement:.1f}%")
    
    if 'fixed_n' in experiment.results:
        errors = experiment.results['fixed_n']['errors']
        k_values = experiment.results['fixed_n']['k_values']
        
        print("\n固定本地样本量n时:")
        for i, k in enumerate(k_values):
            avg_error = errors['average'][i]
            csl_error = errors['csl_one_step'][i]
            improvement = (avg_error - csl_error) / avg_error * 100
            print(f"  k={k}: 平均误差={avg_error:.4f}, CSL误差={csl_error:.4f}, 改进={improvement:.1f}%")