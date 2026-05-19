import numpy as np

def plot_metrics_lineplot(results_dict, metrics, k_values):
    # Set up the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Define colors for different models
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    # Create line plots for each metric
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3

        # Tìm giá trị cao nhất cho mỗi K
        max_values = []
        for k_idx in range(len(k_values)):
            max_val = max(results[metric][k_idx] for results in results_dict.values())
            max_values.append(max_val)

        # Plot each model's results
        for i, (model_name, results) in enumerate(results_dict.items()):
            # Vẽ đường bình thường cho mỗi model
            line = axs[row, col].plot(k_values, results[metric], 'o-',
                                    label=model_name, color=colors[i])

            # Nếu đường này có điểm nào bằng max, vẽ thêm đường nét đứt
            for k_idx, val in enumerate(results[metric]):
                if val == max_values[k_idx]:
                    axs[row, col].plot(k_values[k_idx], val, 'r*', markersize=10)

        # # Vẽ đường nét đứt cho giá trị cao nhất
        # axs[row, col].plot(k_values, max_values, '--', color='red',
        #                   alpha=0.5, label='Best Values')

        # Customize subplot appearance
        axs[row, col].set_title(f'{metric.upper()}@K')
        axs[row, col].set_xlabel('K')
        axs[row, col].set_ylabel(metric.upper())
        axs[row, col].legend()
        axs[row, col].grid(True)

    # Remove empty subplots if there are fewer than 6 metrics
    if len(metrics) < 6:
        for i in range(len(metrics), 6):
            row = i // 3
            col = i % 3
            fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Chuyển data sang dạng DataFrame để dễ vẽ
def plot_metrics_boxplot(results_dict, metrics, k_values):
    plt.figure(figsize=(15, 10))

    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)

        # Tạo data cho boxplot
        data = []
        labels = []
        for model_name, results in results_dict.items():
            data.append(results[metric])
            labels.extend([model_name] * len(k_values))

        df = pd.DataFrame({
            'Model': labels,
            f'{metric.upper()}@K': [val for model_vals in data for val in model_vals],
            'K': k_values * len(results_dict)
        })

        # Vẽ boxplot
        sns.boxplot(data=df, x='Model', y=f'{metric.upper()}@K')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric.upper()}@K Distribution')

    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(results_dict, metrics, k_values):
    for metric in metrics:
        # Tạo matrix cho heatmap
        data = []
        for model_name, results in results_dict.items():
            data.append(results[metric])

        plt.figure(figsize=(10, 8))
        sns.heatmap(data,
                   annot=True,
                   fmt='.4f',
                   xticklabels=[f'K={k}' for k in k_values],
                   yticklabels=list(results_dict.keys()),
                   cmap='YlOrRd')
        plt.title(f'{metric.upper()}@K Comparison')
        plt.tight_layout()
        plt.show()

def plot_metrics_barplot(results_dict, metrics, k_values):
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        x = np.arange(len(results_dict))
        width = 0.2

        for i, k in enumerate(k_values):
            values = [results[metric][i] for results in results_dict.values()]
            plt.bar(x + i*width, values, width, label=f'K={k}')

        plt.xlabel('Models')
        plt.ylabel(f'{metric.upper()}@K')
        plt.title(f'{metric.upper()}@K Comparison')
        plt.xticks(x + width*1.5, list(results_dict.keys()), rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

def compute_statistics(results_dict, metrics):
    stats = {}
    for metric in metrics:
        stats[metric] = {
            'best_model': max(results_dict.items(),
                            key=lambda x: sum(x[1][metric])/len(x[1][metric]))[0],
            'mean_values': {model: sum(results[metric])/len(results[metric])
                          for model, results in results_dict.items()},
            'rankings': {k: sorted(results_dict.keys(),
                                 key=lambda x: results_dict[x][metric][i],
                                 reverse=True)
                        for i, k in enumerate([1, 5, 10, 20])}
        }
    return stats



def plot_radar_chart(results_dict, metrics):
    # Tính giá trị trung bình cho mỗi metric
    avg_metrics = {model: {metric: sum(results[metric])/len(results[metric])
                          for metric in metrics}
                  for model, results in results_dict.items()}

    # Vẽ radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for model, metrics_values in avg_metrics.items():
        values = [metrics_values[m] for m in metrics]
        values += values[:1]
        angles_plot = np.concatenate((angles, [angles[0]]))

        ax.plot(angles_plot, values, 'o-', linewidth=2, label=model)
        ax.fill(angles_plot, values, alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Overall Model Performance")
    plt.show()


# Gọi các hàm vẽ
metrics = ['recall', 'precision', 'ndcg']
k_values = [1, 5, 10, 20]

plot_metrics_boxplot(results_dict, metrics, k_values)


##########
results_dict = {}


# Light GCN standard
add_results(
    results_dict,
    model_name='LightGCN(standard)',
    recall=[0.00104, 0.00114, 0.00233, 0.00318],
    precision=[0.00440, 0.00110, 0.00121, 0.00083],
    ndcg=[0.00440, 0.00201, 0.00237, 0.00264],
    hit_ratio=[0.00440, 0.00468, 0.00540, 0.00569],
    mrr=[0.00440, 0.00550, 0.01210, 0.01650]
)
