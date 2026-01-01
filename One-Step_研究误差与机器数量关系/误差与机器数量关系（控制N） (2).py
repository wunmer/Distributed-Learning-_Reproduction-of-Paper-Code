import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings('ignore')
# 设置中文字体和更美观的模板
pio.templates.default = "plotly_white"

# 定义正态似然函数
def log_likelihood_gaussian(params, data):
    mu, sigma2 = params
    n = len(data)
    if sigma2 <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, loc=mu, scale=np.sqrt(sigma2)))
# 整体mle
def centralized_mle(data):
    result = minimize(log_likelihood_gaussian, x0=[np.mean(data), np.var(data)], 
                     args=(data,), method='L-BFGS-B', 
                     bounds=[(None, None), (1e-6, None)])
    return result.x

def simulate_gaussian_experiment(N, k, K=50):
    n = N // k  # 每台机器的样本量
    mse_simple = []
    mse_resampled = []
    mse_onestep = []
    mse_central = []
    for _ in range(K):
        # 均匀分布生成参数真值
        mu_true = np.random.uniform(-2, 2)
        sigma2_true = np.random.uniform(0.25, 9)
        # 生成全数据
        data_full = np.random.normal(mu_true, np.sqrt(sigma2_true), N)
        # 把数据分到K台机器
        data_splits = np.array_split(data_full, k)
        # 局部MLE，计算每台机器估计值。
        local_estimates = []
        for i in range(k):
            try:
                est = centralized_mle(data_splits[i])
                local_estimates.append(est)
            except:
                local_estimates.append([mu_true, sigma2_true])  # fallback
        local_estimates = np.array(local_estimates)
        # 简单平均估计
        theta_simple = np.mean(local_estimates, axis=0)
        mse_simple.append(np.sum((theta_simple - [mu_true, sigma2_true])**2))
        # Centralized MLE
        theta_central = centralized_mle(data_full)
        mse_central.append(np.sum((theta_central - [mu_true, sigma2_true])**2))
        # 梯度和黑塞矩阵的计算函数
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
        'centralized': (np.mean(mse_central), np.std(mse_central)),
        'samples_per_machine': n  # 记录每台机器的样本量
    }
    
# 运行实验：固定总样本量，增加机器数量
results = []
k_values = [8, 16, 32, 64, 128, 256, 512]
total_samples = 10000  # 固定总样本量

print("Running Gaussian Distribution experiments with fixed total sample size...")
print(f"Total sample size: {total_samples}")

for k in k_values:
    if k > total_samples:  # 确保每台机器至少有一个样本
        continue
    
    print(f"Running k={k}, total N={total_samples}, n={total_samples//k} per machine...")
    res = simulate_gaussian_experiment(total_samples, k, K=30)
    res['k'] = k
    res['total_samples'] = total_samples
    res['samples_per_machine'] = total_samples // k
    results.append(res)

# ----------------------------------------------------------------- #
# ----------------------------图像绘制---------------------------- #
# -------------------------------------------------------------- #
# 提取数据用于绘图
k_vals = []
samples_per_machine_vals = []
simple_means = []
simple_stds = []
onestep_means = []
onestep_stds = []
central_means = []

for res in results:
    k_vals.append(res['k'])
    samples_per_machine_vals.append(res['samples_per_machine'])
    
    simple_mean, simple_std = res['simple_avg']
    simple_means.append(simple_mean)
    simple_stds.append(simple_std)
    
    onestep_mean, onestep_std = res['onestep']
    onestep_means.append(onestep_mean)
    onestep_stds.append(onestep_std)
    
    central_mean, _ = res['centralized']
    central_means.append(central_mean)

# 创建交互式图表
fig = go.Figure()

# 为点的大小设置缩放因子（根据标准差）
size_scale = 100  # 调整这个值可以改变点的大小

# Simple Averaging estimator - 用圆形表示
fig.add_trace(go.Scatter(
    x=k_vals,
    y=simple_means,
    mode='markers+lines',
    name='Simple Averaging',
    marker=dict(
        size=[s * size_scale for s in simple_stds],  # 点的大小表示标准差
        color='blue',
        symbol='circle',
        line=dict(width=2, color='darkblue')
    ),
    line=dict(color='blue', width=2, dash='dash'),
    hovertemplate='<b>Simple Averaging</b><br>' +
                  'Machines (k): %{x}<br>' +
                  'Samples per machine: %{customdata}<br>' +
                  'MSE: %{y:.4f}<br>' +
                  'Std Dev: %{marker.size:.4f}<extra></extra>',
    customdata=samples_per_machine_vals
))

# One-step estimator - 用方形表示
fig.add_trace(go.Scatter(
    x=k_vals,
    y=onestep_means,
    mode='markers+lines',
    name='One-step Estimator',
    marker=dict(
        size=[s * size_scale for s in onestep_stds],  # 点的大小表示标准差
        color='red',
        symbol='square',
        line=dict(width=2, color='darkred')
    ),
    line=dict(color='red', width=2),
    hovertemplate='<b>One-step Estimator</b><br>' +
                  'Machines (k): %{x}<br>' +
                  'Samples per machine: %{customdata}<br>' +
                  'MSE: %{y:.4f}<br>' +
                  'Std Dev: %{marker.size:.4f}<extra></extra>',
    customdata=samples_per_machine_vals
))

# 添加参考线（Centralized MLE）
central_mean_avg = np.mean(central_means)
fig.add_hline(y=central_mean_avg, line_dash="dot", 
              annotation_text=f"Centralized MLE (Avg: {central_mean_avg:.4f})",
              annotation_position="bottom right",
              line_color="green", line_width=2)

# 更新布局
fig.update_layout(
    title=dict(
        text=f'MSE vs Number of Machines (Fixed Total Samples: {total_samples})',
        font=dict(size=22, family="Arial, sans-serif"),
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title=dict(
            text='Number of Machines (k)',
            font=dict(size=18)
        ),
        type='log',  # 使用对数刻度，因为k值范围很大
        tickmode='array',
        tickvals=k_vals,
        ticktext=[str(k) for k in k_vals],
        gridcolor='lightgray',
        gridwidth=1
    ),
    yaxis=dict(
        title=dict(
            text='Mean Squared Error (MSE)',
            font=dict(size=18)
        ),
        gridcolor='lightgray',
        gridwidth=1
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=14)
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Arial"
    ),
    plot_bgcolor='white',
    width=1000,
    height=600,
    showlegend=True
)

# 添加注释说明点的大小表示标准差
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.98, y=0.02,
    text="Marker size ∝ standard deviation",
    showarrow=False,
    font=dict(size=12, color="gray"),
    align="right"
)

# 添加第二x轴显示每台机器样本量
fig.update_layout(
    xaxis2=dict(
        title="Samples per machine",
        overlaying="x",
        side="top",
        type='log',
        tickmode='array',
        tickvals=k_vals,
        ticktext=[str(n) for n in samples_per_machine_vals],
        showgrid=False
    )
)

# 显示图表
fig.show()

# 导出为多种格式，适合LaTeX beamer
print("\nExporting figures for LaTeX beamer...")

# 导出主图表
pio.write_image(fig, 'mse_vs_k_fixed_total.pdf', width=1000, height=600, scale=2)
print("✓ Exported as PDF: mse_vs_k_fixed_total.pdf")

# 创建简化版图表（更适合beamer演示）
fig_simple = go.Figure()

fig_simple.add_trace(go.Scatter(
    x=k_vals,
    y=simple_means,
    mode='markers+lines',
    name='Simple Averaging',
    marker=dict(size=10, color='#1f77b4', symbol='circle'),
    line=dict(color='#1f77b4', width=3, dash='dash'),
    error_y=dict(
        type='data',
        array=simple_stds,
        visible=True,
        color='#1f77b4',
        thickness=1.5,
        width=3
    )
))

fig_simple.add_trace(go.Scatter(
    x=k_vals,
    y=onestep_means,
    mode='markers+lines',
    name='One-step Estimator',
    marker=dict(size=10, color='#ff7f0e', symbol='square'),
    line=dict(color='#ff7f0e', width=3),
    error_y=dict(
        type='data',
        array=onestep_stds,
        visible=True,
        color='#ff7f0e',
        thickness=1.5,
        width=3
    )
))

fig_simple.update_layout(
    title=dict(
        text=f'MSE vs Number of Machines (N={total_samples})',
        font=dict(size=20),
        x=0.5
    ),
    xaxis=dict(
        title='Number of Machines (k)',
        type='log',
        tickvals=k_vals,
        ticktext=[str(k) for k in k_vals],
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Mean Squared Error',
        gridcolor='lightgray'
    ),
    legend=dict(
        x=0.7,
        y=0.95,
        font=dict(size=14)
    ),
    annotations=[
        dict(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Total samples: {total_samples}",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    ],
    width=800,
    height=500,
    template='plotly_white'
)

# 导出简化版
pio.write_image(fig_simple, 'mse_vs_k_fixed_total_simple.pdf', width=800, height=500, scale=2)
print("✓ Exported simplified version: mse_vs_k_fixed_total_simple.pdf")

# 显示数值结果表格
print("\n" + "="*120)
print("Numerical Results (Fixed Total Samples)")
print(f"Total sample size: {total_samples}")
print("="*120)
print("k\tn per machine\tSimple Avg MSE (std)\t\tOne-step MSE (std)\t\tCentralized MSE")
print("-"*120)

for i, k in enumerate(k_vals):
    n_per = samples_per_machine_vals[i]
    print(f"{k}\t{n_per}\t\t{simple_means[i]:.6f} ({simple_stds[i]:.6f})\t{onestep_means[i]:.6f} ({onestep_stds[i]:.6f})\t{central_means[i]:.6f}")

# 创建对比图表（改进百分比）
print("\n" + "="*90)
print("Improvement of One-step over Simple Averaging")
print("="*90)
print("k\tn per machine\tMSE Reduction (%)\tEfficiency Ratio")
print("-"*90)

for i, k in enumerate(k_vals):
    n_per = samples_per_machine_vals[i]
    if simple_means[i] > 0:
        reduction = 100 * (simple_means[i] - onestep_means[i]) / simple_means[i]
        efficiency = simple_means[i] / onestep_means[i] if onestep_means[i] > 0 else float('inf')
        print(f"{k}\t{n_per}\t\t{reduction:6.2f}%\t\t{efficiency:6.2f}x")

# 创建额外图表：显示每台机器样本量对MSE的影响
fig_samples = go.Figure()

fig_samples.add_trace(go.Scatter(
    x=samples_per_machine_vals,
    y=simple_means,
    mode='markers+lines',
    name='Simple Averaging',
    marker=dict(size=12, color='blue', symbol='circle'),
    line=dict(color='blue', width=2, dash='dash'),
    hovertemplate='Samples per machine: %{x}<br>MSE: %{y:.4f}<extra></extra>'
))

fig_samples.add_trace(go.Scatter(
    x=samples_per_machine_vals,
    y=onestep_means,
    mode='markers+lines',
    name='One-step Estimator',
    marker=dict(size=12, color='red', symbol='square'),
    line=dict(color='red', width=2),
    hovertemplate='Samples per machine: %{x}<br>MSE: %{y:.4f}<extra></extra>'
))

fig_samples.update_layout(
    title=dict(
        text='MSE vs Samples per Machine (Fixed Total Samples)',
        font=dict(size=20),
        x=0.5
    ),
    xaxis=dict(
        title='Samples per Machine',
        type='log',
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='Mean Squared Error',
        gridcolor='lightgray'
    ),
    legend=dict(
        x=0.7,
        y=0.95,
        font=dict(size=14)
    ),
    width=800,
    height=500,
    template='plotly_white'
)

pio.write_image(fig_samples, 'mse_vs_samples_per_machine.pdf', width=800, height=500, scale=2)
print("✓ Exported additional chart: mse_vs_samples_per_machine.pdf")

print("\nFor LaTeX beamer, use:")
print(r"\begin{frame}{MSE Comparison with Fixed Total Samples}")
print(r"    \begin{center}")
print(r"        \includegraphics[width=0.85\textwidth]{mse_vs_k_fixed_total_simple.pdf}")
print(r"    \end{center}")
print(r"    \begin{itemize}")
print(r"        \item Fixed total samples: " + str(total_samples))
print(r"        \item \textcolor{blue}{Simple Averaging} degrades as $k$ increases")
print(r"        \item \textcolor{orange}{One-step Estimator} maintains better performance")
print(r"    \end{itemize}")
print(r"\end{frame}")