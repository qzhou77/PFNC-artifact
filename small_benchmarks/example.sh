
# - GAS
env CUDA_VISIBLE_DEVICES=1 python main.py model=gcn dataset=cora root=DATASET device=0 | tee gcn-cora-gas.log

# - GAS + PFNC
env CUDA_VISIBLE_DEVICES=1 python main.py model=gcn dataset=cora root=DATASET device=0 pfn=true +model.params.Cora.architecture.grad_add=true | tee gcn-cora-pfn-grad_add.log

# small_benchmarks/GAS vs GAS+PFNC (Cora).png
