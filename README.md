# PFNC

## Meaning of Folder Structure

- utils

  Utility code.

- motivation

  Code for exploratory data collection, such as in Figures 1, 3, 5, and 6.
  
  Figure 1: `motivation/node_classification_inspect_embedding_importance.py`
  
  Figure 3: `motivation/node_classification_inspect_embedding_importance.py`
  
  Figure 5: `motivation/neighbor_num_plot.py`
  
  Figure 6: `motivation/plot_cache_filter_importance.py`, `motivation/node_classification_inspect_embedding_importance.py`

- pfnc-pkg

  Contains the prototype implementation of PFNC.

- csrc, torch_geometric_autoscale, setup.py

  Code adapted from GAS and the model part of torch_geometric_autoscale for PFNC.

- small_benchmarks

  Experimental code and configuration files for small-scale datasets.

- large_benchmarks

  Experimental code and configuration files for large-scale datasets.

## Code Execution Example

```shell
cd small_benchmarks

# - GAS
env CUDA_VISIBLE_DEVICES=1 python main.py model=gcn dataset=cora root=DATASET device=0 | tee gcn-cora-gas.log

# - GAS + PFNC
env CUDA_VISIBLE_DEVICES=1 python main.py model=gcn dataset=cora root=DATASET device=0 pfn=true +model.params.Cora.architecture.grad_add=true | tee gcn-cora-pfn-grad_add.log
```
