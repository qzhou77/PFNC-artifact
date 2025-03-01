import dgl
import torch
import matplotlib.pyplot as plt
from ogb.nodeproppred import DglNodePropPredDataset
import pickle

DIR = ""


# 加载ogbn-arxiv数据集
dataset = DglNodePropPredDataset(name='ogbn-arxiv')
graph = dataset[0][0]

# 获取训练集、验证集和测试集的节点索引
split_idx = dataset.get_idx_split()
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

# 计算每个节点的邻居数量
neighbor_counts = []
for i in range(graph.num_nodes()):
    neighbor_counts.append(graph.in_degree(i) + graph.out_degree(i))

# 将邻居数量转换为张量
neighbor_counts = torch.tensor(neighbor_counts)

output_dir = f"{DIR}/cache_filter_importance/"
# 定义一个函数来绘制邻居数量分布
def plot_neighbor_distribution(node_indices, title, filename):
    # 获取指定节点的邻居数量
    selected_counts = neighbor_counts[node_indices]
    # 对邻居数量进行排序
    sorted_counts, _ = torch.sort(selected_counts, descending=True)
    sorted_counts = sorted_counts.numpy()

    # 绘制排序后的邻居数量
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_counts)

    thresholds = [5, 10, 15, 20, 25]
    for threshold in thresholds:
        # 找到首次小于阈值的索引
        first_index = next((i for i, count in enumerate(sorted_counts) if count <= threshold), None)
        if first_index is not None:
            # 突出显示该点，并添加注释
            plt.scatter(first_index, sorted_counts[first_index], color='red', s=100)
            plt.text(first_index, sorted_counts[first_index], f'$\leq$ {threshold}', fontsize=12, ha='center', va='bottom', rotation=45)

    # 纵坐标log尺度
    plt.yscale('log')

    plt.title(title, fontsize=14)
    plt.xlabel('Node Index (Sorted by Neighbor Count)', fontsize=14)
    plt.ylabel('Number of Neighbors', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir + filename)
    plt.close()

# 绘制整个图的邻居数量分布
plot_neighbor_distribution(torch.arange(graph.num_nodes()), 'Neighbor Distribution', 'whole_graph_neighbor_distribution.png')

# 绘制训练集的邻居数量分布
plot_neighbor_distribution(train_idx, 'Neighbor Distribution of the Training Set', 'train_set_neighbor_distribution.png')

# 绘制验证集的邻居数量分布
plot_neighbor_distribution(valid_idx, 'Neighbor Distribution of the Validation Set', 'valid_set_neighbor_distribution.png')

# 绘制测试集的邻居数量分布
plot_neighbor_distribution(test_idx, 'Neighbor Distribution of the Test Set', 'test_set_neighbor_distribution.png')

# 存储邻居数量最大的10%的节点的node id
num_nodes = graph.num_nodes()
top_10_percent = int(num_nodes * 0.1)
# 获取邻居数量最大的10%的节点的索引 以及对应的邻居数量
top_values, top_indices = torch.topk(neighbor_counts, top_10_percent)


# 将结果保存为pkl文件
pkl_file_path = output_dir + 'top_10_percent_nodes.pkl'
with open(pkl_file_path, 'wb') as f:
    pickle.dump(top_indices, f)

num_pkl_file_path = output_dir + 'top_10_percent_nodes_num.pkl'
with open(num_pkl_file_path, 'wb') as f:
    pickle.dump(top_values, f)

# save all neighbor counts
all_neighbor_counts_pkl_file_path = output_dir + 'all_neighbor_counts.pkl'
with open(all_neighbor_counts_pkl_file_path, 'wb') as f:
    pickle.dump(neighbor_counts, f)

print(f"Top 10% nodes saved to {pkl_file_path}")
print("Plots saved successfully.")