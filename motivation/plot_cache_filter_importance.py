
DIR = ""

import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

all_neighbor_counts_file = f"{DIR}/cache_filter_importance/all_neighbor_counts.pkl"
with open(all_neighbor_counts_file, "rb") as f:
    all_neighbor_counts = pickle.load(f)

def plot_y_hat_and_gradients(fanouts, bs, total_iter_num_current_run, iter_id_list, reorder=0, batch_wise=False, batch_wise_size=16, batch_wise_type="mean"):
    # 生成文件路径模式
    y_hat_file_pattern = f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{}-bs{}/iter-{}_embedding_variance.pkl"
    y_hat_gradients_file_pattern = f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{}-bs{}/iter-{}_embedding_gradients_norm.pkl"
    output_nodes_file_pattern = f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{}-bs{}/iter-{}_output_nodes.pkl"

    # 将 fanouts 转换为字符串
    str_fanouts = "_".join([str(f) for f in fanouts])

    ten_trys_y_hat = []
    ten_trys_y_hat_gradients = []
    all_neighbor_count = []
    for iter_id in iter_id_list:
        # 生成具体的文件路径
        y_hat_file = y_hat_file_pattern.format(str_fanouts, bs, iter_id)
        y_hat_gradients_file = y_hat_gradients_file_pattern.format(str_fanouts, bs, iter_id)

        # 加载数据
        with open(y_hat_file, "rb") as f:
            one_ten_trys_y_hat = pickle.load(f).cpu().numpy()
        with open(y_hat_gradients_file, "rb") as f:
            # 修正拼写错误
            one_ten_trys_y_hat_gradients = pickle.load(f).cpu().numpy()
        with open(output_nodes_file_pattern.format(str_fanouts, bs, iter_id), "rb") as f:
            output_nodes = pickle.load(f)
        neighbor_count = all_neighbor_counts[output_nodes].cpu().numpy()
        print(f"shape of neighbor_count: {neighbor_count.shape}")

        ten_trys_y_hat.append(one_ten_trys_y_hat)
        ten_trys_y_hat_gradients.append(one_ten_trys_y_hat_gradients)
        all_neighbor_count.append(neighbor_count)

    # cat 数据
    ten_trys_y_hat = np.concatenate(ten_trys_y_hat)
    ten_trys_y_hat_gradients_ = np.concatenate(ten_trys_y_hat_gradients)
    all_neighbor_count = np.concatenate(all_neighbor_count)

    if reorder>0:
        # 按照 gradient 排序，获取排序索引，
        if reorder == 1:
            sorted_idx = np.argsort(ten_trys_y_hat_gradients_)
        elif reorder == 2:
            sorted_idx = np.argsort(ten_trys_y_hat)
        elif reorder == 3:
            # by neighbor count, 从小到大
            sorted_idx = np.argsort(all_neighbor_count)
        print(f"shape of ten_trys_y_hat: {ten_trys_y_hat.shape}")
        print(f"shape of ten_trys_y_hat_gradients: {ten_trys_y_hat_gradients_.shape}")
        print(f"shape of sorted_idx: {sorted_idx.shape}")
        ten_trys_y_hat = ten_trys_y_hat[sorted_idx]
        ten_trys_y_hat_gradients_ = ten_trys_y_hat_gradients_[sorted_idx]
        sorted_neighbor_count = all_neighbor_count[sorted_idx]

    if batch_wise:
        # avg_size = bs//batch_wise_size
        avg_size = batch_wise_size
        if batch_wise_type == "mean":
            # 每bs个数据取平均
            ten_trys_y_hat = ten_trys_y_hat.reshape(-1, avg_size).mean(axis=1)
            ten_trys_y_hat_gradients = ten_trys_y_hat_gradients_.reshape(-1, avg_size).mean(axis=1)
            sorted_neighbor_count = sorted_neighbor_count.reshape(-1, avg_size).mean(axis=1)
            var_y_hat_gradients = np.sqrt(ten_trys_y_hat_gradients_.reshape(-1, avg_size).var(axis=1))
        elif batch_wise_type == "max":
            # 每bs个数据取最大
            ten_trys_y_hat = ten_trys_y_hat.reshape(-1, avg_size).max(axis=1)
            ten_trys_y_hat_gradients = ten_trys_y_hat_gradients.reshape(-1, avg_size).max(axis=1)
        elif batch_wise_type == "min":
            # 每bs个数据取最小
            ten_trys_y_hat = ten_trys_y_hat.reshape(-1, avg_size).min(axis=1)
            ten_trys_y_hat_gradients = ten_trys_y_hat_gradients.reshape(-1, avg_size).min(axis=1)
        elif batch_wise_type == "0.xx":
            # 每bs个数据取0.75分位数
            ten_trys_y_hat = np.quantile(ten_trys_y_hat.reshape(-1, avg_size), 0.9, axis=1)
            ten_trys_y_hat_gradients = np.quantile(ten_trys_y_hat_gradients.reshape(-1, avg_size), 0.9, axis=1)
    else:
        ten_trys_y_hat_gradients = ten_trys_y_hat_gradients_
    
    # ten_trys_y_hat = np.log(ten_trys_y_hat)
    # ten_trys_y_hat_gradients = np.sqrt(ten_trys_y_hat_gradients)

    # 打印数据形状
    print(f"shape of ten_trys_y_hat: {ten_trys_y_hat.shape}")
    print(f"shape of ten_trys_y_hat_gradients: {ten_trys_y_hat_gradients.shape}")

    # 创建双Y轴绘图
    fig, ax1 = plt.subplots(figsize=(6.4, 4))

    # 绘制 y_hat 数据
    color = 'tab:blue'
    ax1.set_xlabel('Node Index (sorted by embedding variance)', fontsize=12)
    ax1.set_ylabel('Embedding Variance', color=color, fontsize=12)
    # log scale
    # ax1.set_yscale('log')
    ax1.plot(ten_trys_y_hat, color=color, label="Embedding Variance")
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个Y轴
    ax2 = ax1.twinx()

    # 绘制 y_hat_gradients 数据
    color = 'tab:red'
    ax2.set_ylabel('Embedding Gradient Norm', color=color, fontsize=12)
    ax2.plot(ten_trys_y_hat_gradients, color=color, label="Embedding Gradient Norm")
    # # plot +var
    # ax2.fill_between(range(len(ten_trys_y_hat_gradients)), ten_trys_y_hat_gradients, ten_trys_y_hat_gradients+var_y_hat_gradients, color='red', alpha=0.1)
    # # plot -var
    # ax2.fill_between(range(len(ten_trys_y_hat_gradients)), ten_trys_y_hat_gradients, ten_trys_y_hat_gradients-var_y_hat_gradients, color='red', alpha=0.1)
    
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    # 显示图形
    plt.savefig("y_hat_and_gradients.png")

# 定义参数
fanouts = [5, 5, 5]
bs = 1024
total_iter_num_current_run = 444
# taget_iter_ids = list(range(0, 400, 1))
taget_iter_ids = list(range(0, 400, 1))

plot_y_hat_and_gradients(fanouts, bs, total_iter_num_current_run, taget_iter_ids, reorder=2, batch_wise=True, batch_wise_size=64, batch_wise_type="mean")

# plot_y_hat_and_gradients(fanouts, bs, total_iter_num_current_run, taget_iter_ids, reorder=3, batch_wise=True, batch_wise_size=16, batch_wise_type="mean")

# import time
# for id in taget_iter_ids:
#     # 调用函数进行绘图
#     plot_y_hat_and_gradients(fanouts, bs, total_iter_num_current_run, [id], reorder=3, batch_wise=False)
#     time.sleep(1)