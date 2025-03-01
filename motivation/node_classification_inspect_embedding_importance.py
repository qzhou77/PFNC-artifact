import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from pfn_pkg import PFN

DIR = ""

last_layer_gradients = []
class SAGE(nn.Module):
    def post_init(self, args):
        print(f"[DEBUG] args: {vars(args)}")
        if args.pfn:
            PFN_cfg = {
                **vars(args),
            }
            # print PFN config
            print(f"[DEBUG] PFN config: {PFN_cfg}")
            # if PFN_cfg doesn't have num_nodes, num_layers, assign from self; otherwise check if the values are the same
            if 'num_nodes' not in PFN_cfg:
                PFN_cfg['num_nodes'] = self.num_nodes
            else:
                assert PFN_cfg['num_nodes'] == self.num_nodes, "PFN_cfg['num_nodes'] is not the same as self.num_nodes"
            if 'num_layers' not in PFN_cfg:
                PFN_cfg['num_layers'] = self.num_layers
            else:
                assert PFN_cfg['num_layers'] == self.num_layers, "PFN_cfg['num_layers'] is not the same as self.num_layers"
            self.pfn = PFN(**PFN_cfg)
            l_start = PFN_cfg['start_from']
            l_end = PFN_cfg['end_at']
            assert l_start<self.num_layers and l_start<l_end, "PFN_cfg is not None, but no layer is tracked"
            self.pfn.generate_meta(self.convs[l_start:l_end], [f"conv_{i}" for i in range(l_start, l_end)], [i for i in range(l_start, l_end)])
            self.pfn.init_all_history_queues()

        else:
            self.pfn = None

        if self.pfn is not None:
            print(f"[INFO] PFN is not None")
        else:
            print(f"[INFO] PFN is None")
        
    def __init__(self, in_size, hid_size, out_size, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.convs.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.convs.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.convs.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

        assert args.num_layers == len(self.convs), "args.num_layers is not the same as the number of layers in the model"
        self.num_layers = args.num_layers
        self.num_nodes = args.num_nodes

        # print this model
        print(self)

        self.post_init(args)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            h = layer(block, h)
            if l != len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def layer_wise_full_forward(self, g, x):
        # return the output of each layer
        h = x
        outputs = []
        for l, layer in enumerate(self.convs):
            h = layer(g, h)
            if l != len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
            outputs.append(h)
        return outputs
    
    def full_forward(self, g, x):
        h = x
        for l, layer in enumerate(self.convs):
            h = layer(g, h)
            if l != len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        # Conduct layer-wise inference to get all the node embeddings.
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.convs):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.convs) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.convs) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )

def full_infer(device, graph, nids, model, num_classes, train_loss=False):
    model.eval()
    with torch.no_grad():
        preds = model.full_forward(graph.to(device), graph.ndata["feat"].to(device))
        if train_loss:
            train_loss = F.cross_entropy(preds[nids[0]], graph.ndata["label"][nids[0]].to(device))
        if not isinstance(nids, list):
            nids = [nids]
        accs = []
        for nid in nids:
            pred = preds[nid]
            label = graph.ndata["label"][nid].to(pred.device)
            acc = MF.accuracy(
                pred, label, task="multiclass", num_classes=num_classes
            )
            accs.append(acc)
        if train_loss:
            return accs, train_loss.item()
        else:
            return accs



def create_ordered_train_loaders(num_loaders, shuffle_train_idx, args, device, g, dataset):
    """
    Function to create multiple ordered train loaders based on the given number and shuffle setting.

    Args:
        num_loaders (int): Number of loaders to generate.
        shuffle_train_idx (bool): Whether to shuffle train indices.
        args: Argument containing batch size, fanouts, etc.
        device: Device to load the data on.
        g: Graph for the sampler.
        dataset: Dataset containing train and validation indices.

    Returns:
        List of ordered train loaders.
    """
    # Get train indices and move to the proper device
    train_idx = dataset.train_idx.to(device)
    
    # Shuffle the train_idx if specified
    if shuffle_train_idx:
        train_idx = train_idx[torch.randperm(train_idx.shape[0])]

    # Create the samplers for each loader
    samplers = [
        NeighborSampler(
            args.fanouts,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        for _ in range(num_loaders)
    ]

    # Determine if using UVA mode
    use_uva = args.mode == "mixed"

    # Generate the ordered train loaders
    train_loaders = [
        DataLoader(
            g,
            train_idx,
            samplers[i],
            device=device,
            batch_size=args.bs,
            shuffle=False,  # shuffle is handled at the dataset level
            drop_last=False,
            num_workers=0,
            use_uva=use_uva,
        )
        for i in range(num_loaders)
    ]
    
    return train_loaders

def train_with_K_neighbor_sampling_try(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        args.fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"

    # train_dataloader = DataLoader(
    #     g,
    #     train_idx,
    #     sampler,
    #     device=device,
    #     batch_size=args.bs,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     use_uva=use_uva,
    # )

    # K
    sample_size_for_reducing_variance = args.var_try

    results = []

    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            train_idx = train_idx[torch.randperm(train_idx.shape[0])]
            train_dataloaders = create_ordered_train_loaders(sample_size_for_reducing_variance, False, args, device, g, dataset)
            len_of_train_dataloader = len(train_dataloaders[0])
            train_dataloaders = [iter(lder) for lder in train_dataloaders]
            
            for i_iter in range(len_of_train_dataloader):
                opt.zero_grad()
                for i_loader, train_dataloader in enumerate(train_dataloaders):
                    input_nodes, output_nodes, blocks = next(train_dataloader)
                    x = blocks[0].srcdata["feat"]
                    y = blocks[-1].dstdata["label"]
                    y_hat = model(blocks, x)
                    loss = F.cross_entropy(y_hat, y)
                    loss.backward()
                    total_loss += loss.item()
                # scale the gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= sample_size_for_reducing_variance
                opt.step()

            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )


class ReplaceGradHook:
    def __init__(self, replacement):
        self.replacement = replacement.requires_grad_(True)
        # retain_grad
        # self.replacement.retain_grad()
        self.original = None

    def hook_forward(self, module, inputs, output):
        # 保存原始输出（detach），返回替换后的特征
        # print(f"inputs: {inputs}")
        # print(f"output: {output}")
        self.original = output
        self.original.retain_grad()
        return self.replacement  # 直接返回替换特征，梯度将基于它计算

    def hook_backward(self, grad):
        # 将替换特征的梯度复制到原始输出的梯度
        # print(f"hook_backward: grad={grad.norm().item():.4f}, shape={grad.shape}")
        if self.original.grad is None:
            self.original.grad = grad.clone()
        else:
            self.original.grad += grad.clone()
            print(f"impossible")
        return grad
    
def train_with_exact_node_embedding(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        args.fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    full_neighbor_sampler = NeighborSampler(
        [-1, -1, -1],  # full neighbor sampling
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    # assert queue_size is same as the number of batches in the dataloader
    # print(f"[DEBUG] len(dataloader): {len(train_dataloader)}")

    results = []
    module_name_list = ['convs.0', 'convs.1', 'convs.2']
    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0
        for epoch in range(args.epochs):

            # full_neighbor_dataloader = DataLoader(
            #     g,
            #     train_idx,
            #     full_neighbor_sampler,
            #     device=device,
            #     batch_size=args.bs,
            #     shuffle=False,
            #     drop_last=False,
            #     num_workers=0,
            #     use_uva=use_uva,
            # )
            # iter_full_neighbor_dataloader = iter(full_neighbor_dataloader)
            total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]

                # 生成替换值（精确节点嵌入，使用full neighbor sampling）
                with torch.no_grad():
                    # out1 = torch.randn(4, 5)
                    # out2 = torch.randn(4, 2)
                    # out3 = torch.randn(4, 1)
                    all_out1, all_out2, all_out3 = model.layer_wise_full_forward(g, g.ndata["feat"])
                    replacements = {
                        'convs.0': all_out1[blocks[0].dstdata['_ID']],
                        'convs.1': all_out2[blocks[1].dstdata['_ID']],
                        'convs.2': all_out3[blocks[2].dstdata['_ID']]
                    }

                model.train()

                # 注册前向和反向钩子
                hooks = []
                grad_hooks = []
                hook_objs = []
                for name, module in model.named_modules():
                    if name in module_name_list:
                        hook_obj = ReplaceGradHook(replacements[name])
                        hook_objs.append(hook_obj)
                        # 注册前向钩子
                        forward_hook = module.register_forward_hook(hook_obj.hook_forward)
                        hooks.append(forward_hook)
                        # 注册反向钩子
                        backward_hook = hook_obj.replacement.register_hook(hook_obj.hook_backward)
                        grad_hooks.append(backward_hook)


                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                for hook_obj in hook_objs[::-1]:
                    hook_obj.original.backward(hook_obj.original.grad)

                opt.step()
                total_loss += loss.item()

                # 移除所有钩子
                for hook in hooks:
                    hook.remove()
                for hook in grad_hooks:
                    hook.remove()

            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )


def train_with_exact_embedding_and_variance_reduction(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)


    use_uva = args.mode == "mixed"


    # assert queue_size is same as the number of batches in the dataloader
    # print(f"[DEBUG] len(dataloader): {len(train_dataloader)}")

    sample_size_for_reducing_variance = args.var_try

    results = []
    module_name_list = ['convs.0', 'convs.1', 'convs.2']
    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0
        for epoch in range(args.epochs):

            # full_neighbor_dataloader = DataLoader(
            #     g,
            #     train_idx,
            #     full_neighbor_sampler,
            #     device=device,
            #     batch_size=args.bs,
            #     shuffle=False,
            #     drop_last=False,
            #     num_workers=0,
            #     use_uva=use_uva,
            # )
            # iter_full_neighbor_dataloader = iter(full_neighbor_dataloader)
            total_loss = 0
            train_idx = train_idx[torch.randperm(train_idx.shape[0])]
            train_dataloaders = create_ordered_train_loaders(sample_size_for_reducing_variance, False, args, device, g, dataset)
            len_of_train_dataloader = len(train_dataloaders[0])
            train_dataloaders = [iter(lder) for lder in train_dataloaders]

            for i_iter in range(len_of_train_dataloader):
                opt.zero_grad()
                # 生成替换值（精确节点嵌入，使用full neighbor sampling）
                with torch.no_grad():
                    all_out1, all_out2, all_out3 = model.layer_wise_full_forward(g, g.ndata["feat"])
       
                model.train()
                for i_loader, train_dataloader in enumerate(train_dataloaders):
                    input_nodes, output_nodes, blocks = next(train_dataloader)
                    x = blocks[0].srcdata["feat"]
                    y = blocks[-1].dstdata["label"]
                    
                    # 替换值
                    replacements = {
                        'convs.0': all_out1[blocks[0].dstdata['_ID']],
                        'convs.1': all_out2[blocks[1].dstdata['_ID']],
                        'convs.2': all_out3[blocks[2].dstdata['_ID']]
                    }

                    # 注册前向和反向钩子
                    hooks = []
                    grad_hooks = []
                    hook_objs = []
                    for name, module in model.named_modules():
                        if name in module_name_list:
                            hook_obj = ReplaceGradHook(replacements[name])
                            hook_objs.append(hook_obj)
                            # 注册前向钩子
                            forward_hook = module.register_forward_hook(hook_obj.hook_forward)
                            hooks.append(forward_hook)
                            # 注册反向钩子
                            backward_hook = hook_obj.replacement.register_hook(hook_obj.hook_backward)
                            grad_hooks.append(backward_hook)


                    y_hat = model(blocks, x)
                    loss = F.cross_entropy(y_hat, y)
                    loss.backward()
                    total_loss += loss.item()
                    for hook_obj in hook_objs[::-1]:
                        hook_obj.original.backward(hook_obj.original.grad)

                    # 移除所有钩子
                    for hook in hooks:
                        hook.remove()
                    for hook in grad_hooks:
                        hook.remove()


                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= sample_size_for_reducing_variance
                opt.step()

            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )


def train_with_logging_embedding_gradients(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        args.fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    # assert queue_size is same as the number of batches in the dataloader
    print(f"[DEBUG] len(dataloader): {len(train_dataloader)}")

    sample_size_for_calculating_variance = 10

    results = []
    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0

        # create hook to save gradient on embedding
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                # # print len of grad_input, and print the type of every element in grad_input, and if it is tensor, print the shape
                # len_grad_input = len(grad_input)
                # print(f"len(grad_input): {len_grad_input}")
                # for i, grad in enumerate(grad_input):
                #     print(f"type(grad_input[{i}]): {type(grad)}")
                #     if isinstance(grad, torch.Tensor):
                #         print(f"grad_input[{i}].shape: {grad.shape}")
                # # print len of grad_output, and print the type of every element in grad_output, and if it is tensor, print the shape
                # len_grad_output = len(grad_output)
                # print(f"len(grad_output): {len_grad_output}")
                # for i, grad in enumerate(grad_output):
                #     print(f"type(grad_output[{i}]): {type(grad)}")
                #     if isinstance(grad, torch.Tensor):
                #         print(f"grad_output[{i}].shape: {grad.shape}")
                
                last_layer_gradients.append(grad_output[0])
            return hook

        total_iter_num_current_run = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                if model.pfn is not None:
                    model.pfn.detached_grad_correction([output_nodes for _ in range(args.num_layers)])
            
                opt.step()
                total_loss += loss.item()

                # complete one batch optimization update, log the gradients

                # use 10 dataloders to calculate the variance of gradients of a random batch, 
                # with different neighbor sampling try. 
                gradient_dataloaders = create_ordered_train_loaders(sample_size_for_calculating_variance, True, args, device, g, dataset)
                # ten_trys_y_hat_gadients = []
                last_layer_gradients = []
                ten_trys_y_hat = []
                for i_loader, gradient_dataloader in enumerate(gradient_dataloaders):
                    for i_batch, (input_nodes, output_nodes, blocks) in enumerate(gradient_dataloader):
                        x = blocks[0].srcdata["feat"]
                        y = blocks[-1].dstdata["label"]

                        # add hook
                        # print(model.convs[2])
                        hook_conv3 = model.convs[2].register_backward_hook(save_gradient("conv3"))

                        y_hat = model(blocks, x)
                        loss = F.cross_entropy(y_hat, y)
                        opt.zero_grad()
                        loss.backward()
                        
                        # remove hook
                        hook_conv3.remove()

                        break # ok, here I break it, so it just use one batch 

                    ten_trys_y_hat.append(y_hat.detach().cpu())
                    ten_trys_y_hat_gadients = last_layer_gradients
                    print(f"len(ten_trys_y_hat_gadients): {len(ten_trys_y_hat_gadients)}")
                
                # stack ten_trys_y_hat and calculate the variance
                ten_trys_y_hat = torch.stack(ten_trys_y_hat)
                print(f"ten_trys_y_hat.shape: {ten_trys_y_hat.shape}")
                ten_trys_y_hat = ten_trys_y_hat.var(dim=0).sum(dim=-1)

                # stack ten_trys_y_hat_gadients and calculate the norm
                ten_trys_y_hat_gadients = torch.stack(ten_trys_y_hat_gadients)
                print(f"ten_trys_y_hat_gadients.shape: {ten_trys_y_hat_gadients.shape}")
                ten_trys_y_hat_gadients = ten_trys_y_hat_gadients.norm(dim=-1).mean(dim=0)
                print(f"ten_trys_y_hat_gadients.shape: {ten_trys_y_hat_gadients.shape}")


                # save the gradients to a file
                import pickle
                # change a list args.fanouts to a string
                str_fanouts = "_".join([str(fanout) for fanout in args.fanouts])
                with open(f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{str_fanouts}-bs{args.bs}/iter-{total_iter_num_current_run}_embedding_variance.pkl", "wb") as f:
                    pickle.dump(ten_trys_y_hat, f)
                with open(f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{str_fanouts}-bs{args.bs}/iter-{total_iter_num_current_run}_embedding_gradients_norm.pkl", "wb") as f:
                    pickle.dump(ten_trys_y_hat_gadients, f)
                # save output_nodes
                with open(f"{DIR}/gradient_variance_and_convergence/last_layer_embeddings_variance-{str_fanouts}-bs{args.bs}/iter-{total_iter_num_current_run}_output_nodes.pkl", "wb") as f:
                    pickle.dump(output_nodes.cpu(), f)

                print(f">>>>>>>>>>>>>>>>> ALREADY SAVED THE NO. {total_iter_num_current_run} ITERATION EMBEDDING VAR and GRADIENTS NORM <<<<<<<<<<<<<<<<<<")
            
                total_iter_num_current_run += 1

                print(f"CURRENT it: {it}, total_iter_num_current_run: {total_iter_num_current_run}")
            # [INFO], run =  1, epoch =  54, loss = 0.82954, train = 0.7563, val = 0.7183, test = 0.7129, best_val = 0.7185, bestval_test = 0.7115
            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            # accs = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
            # print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {0}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )
    



def train_with_logging_gradients(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        args.fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    # assert queue_size is same as the number of batches in the dataloader
    print(f"[DEBUG] len(dataloader): {len(train_dataloader)}")

    sample_size_for_calculating_variance = 10

    results = []
    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0

        total_iter_num_current_run = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                if model.pfn is not None:
                    model.pfn.detached_grad_correction([output_nodes for _ in range(args.num_layers)])
            
                opt.step()
                total_loss += loss.item()

                # complete one batch optimization update, log the gradients

                # use 10 dataloders to calculate the variance of gradients of a random batch, 
                # with different neighbor sampling try. 
                gradient_dataloaders = create_ordered_train_loaders(sample_size_for_calculating_variance, True, args, device, g, dataset)
                all_loaders_all_grads = []
                for i_loader, gradient_dataloader in enumerate(gradient_dataloaders):
                    for i_batch, (input_nodes, output_nodes, blocks) in enumerate(gradient_dataloader):
                        x = blocks[0].srcdata["feat"]
                        y = blocks[-1].dstdata["label"]
                        y_hat = model(blocks, x)
                        loss = F.cross_entropy(y_hat, y)
                        opt.zero_grad()
                        loss.backward()

                        # log the gradients
                        all_grads = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # print(f"[DEBUG] {name} grad: {param.grad}")
                                all_grads[name] = param.grad.cpu()
                            else:
                                print(f"[DEBUG] {name} grad: None")
                        
                        # import pdb;pdb.set_trace()

                        break
                    all_loaders_all_grads.append(all_grads)

                # save the gradients to a file
                import pickle
                # change a list args.fanouts to a string
                str_fanouts = "_".join([str(fanout) for fanout in args.fanouts])
                with open(f"{DIR}/gradient_variance_and_convergence/gradient_data-{str_fanouts}-bs{args.bs}/iter-{total_iter_num_current_run}_gradient_data.pkl", "wb") as f:
                    pickle.dump(all_loaders_all_grads, f)
            
                total_iter_num_current_run += 1

                print(f"CURRENT it: {it}, total_iter_num_current_run: {total_iter_num_current_run}")
            # [INFO], run =  1, epoch =  54, loss = 0.82954, train = 0.7563, val = 0.7183, test = 0.7129, best_val = 0.7185, bestval_test = 0.7115
            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            # accs = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
            # print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {0}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )
    

def test_dataloader_for_logging_gradients(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
        
    ordered_train_loader1, ordered_train_loader2, ordered_train_loader3 = create_ordered_train_loaders(3, True, args, device, g, dataset)

    print_time = 1
    # print and compare batch from different dataloaders
    for i in range(print_time):
        num = 0
        for (input_nodes1, output_nodes1, blocks1), (input_nodes2, output_nodes2, blocks2), (input_nodes3, output_nodes3, blocks3) in zip(ordered_train_loader1, ordered_train_loader2, ordered_train_loader3):
            # print(f"[DEBUG] input_nodes1: {input_nodes1}, \n input_nodes2: {input_nodes2}, \n input_nodes3: {input_nodes3}")
            # print(f"[DEBUG] output_nodes1: {output_nodes1}, \n output_nodes2: {output_nodes2}, \n output_nodes3: {output_nodes3}")
            # print(f"[DEBUG] blocks1: {blocks1}, \n blocks2: {blocks2}, \n blocks3: {blocks3}")
            # break
            b1_layer_1, b2_layer_2, b3_layer_3 = blocks1[0], blocks1[1], blocks1[2]
            b2_layer_1, b2_layer_2, b2_layer_3 = blocks2[0], blocks2[1], blocks2[2]
            b3_layer_1, b3_layer_2, b3_layer_3 = blocks3[0], blocks3[1], blocks3[2]
            # 
            # compare b1_layer_1.srcnodes(), b2_layer_1.srcnodes(), b3_layer_1.srcnodes()
            # regard them as three sets of input nodes, we want to draw the venn diagram
            #
            for layer_i in range(3):
                set1 = set(blocks1[layer_i].srcdata[dgl.NID].tolist())
                set2 = set(blocks2[layer_i].srcdata[dgl.NID].tolist())
                set3 = set(blocks3[layer_i].srcdata[dgl.NID].tolist())

                # 计算基础集合操作
                intersection_all = set1 & set2 & set3
                pairwise_12 = (set1 & set2) - intersection_all
                pairwise_13 = (set1 & set3) - intersection_all
                pairwise_23 = (set2 & set3) - intersection_all
                unique1 = set1 - set2 - set3
                unique2 = set2 - set1 - set3
                unique3 = set3 - set1 - set2

                # 打印统计信息
                print(f"\nBatch {num} 层1源节点分析：")
                print(f"总节点数 | set1: {len(set1)}, set2: {len(set2)}, set3: {len(set3)}")
                print(f"三交集合: {len(intersection_all)}")
                print(f"两两交集 | 1-2: {len(pairwise_12)}, 1-3: {len(pairwise_13)}, 2-3: {len(pairwise_23)}")
                print(f"唯一节点 | 1: {len(unique1)}, 2: {len(unique2)}, 3: {len(unique3)}")

                # 可选：生成维恩图（需要安装matplotlib_venn）
                try:
                    from matplotlib_venn import venn3
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=(10,6))
                    venn3([set1, set2, set3], 
                        ('Loader1', 'Loader2', 'Loader3'))
                    plt.title(f"Batch {num} Layer {layer_i} Source Nodes Venn Diagram")
                    plt.savefig(f"{DIR}/gradient_variance_and_convergence/Batch_{num}_Layer_{layer_i}_Source_Nodes_Venn_Diagram.png")
                except ImportError:
                    print("安装matplotlib-venn包后可显示维恩图")
                import pdb;pdb.set_trace()

            # 
            # check whether output_nodes are exactly the same
            # assert torch.equal(output_nodes1, output_nodes2), "output_nodes1 is not the same as output_nodes2"
            # assert torch.equal(output_nodes1, output_nodes3), "output_nodes1 is not the same as output_nodes3"
            num += 1
            print(f"[DEBUG] {num}th batch is the same")






def train(args, device, g, dataset, model_config, num_classes):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        # [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        args.fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    # val_dataloader = DataLoader(
    #     g,
    #     val_idx,
    #     sampler,
    #     device=device,
    #     batch_size=1024,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,
    #     use_uva=use_uva,
    # )

    # assert queue_size is same as the number of batches in the dataloader
    print(f"[DEBUG] len(dataloader): {len(train_dataloader)}")


    results = []
    for run in range(args.runs):
        model = SAGE(*model_config).to(device)

        # convert model and graph to bfloat16 if needed
        if args.dt == "bfloat16":
            g = dgl.to_bfloat16(g)
            model = model.to(dtype=torch.bfloat16)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = 0
        best_val_test = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for it, (input_nodes, output_nodes, blocks) in enumerate(
                train_dataloader
            ):
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                if model.pfn is not None:
                    model.pfn.detached_grad_correction([output_nodes for _ in range(args.num_layers)])
            
                opt.step()
                total_loss += loss.item()
            # acc = evaluate(model, g, val_dataloader, num_classes)
            # print(
            #     "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
            #         epoch, total_loss / (it + 1), acc.item()
            #     )
            # )

            # [INFO], run =  1, epoch =  54, loss = 0.82954, train = 0.7563, val = 0.7183, test = 0.7129, best_val = 0.7185, bestval_test = 0.7115
            accs, train_loss = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes, train_loss=True)
            # accs = full_infer(device, g, [dataset.train_idx, dataset.val_idx, dataset.test_idx], model, num_classes)
            train_acc, val_acc, test_acc = accs
            if val_acc > best_val:
                best_val = val_acc
                best_val_test = test_acc

            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {float(train_loss):.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
            # print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {0}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val:.4f}, bestval_test = {best_val_test:.4f}")
        results.append(best_val_test)
    results = torch.tensor(results)
    print(
        "Final Results: {:.2f} ± {:.2f}".format(results.mean().item()*100, results.std().item()*100)
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training."
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="dataset name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="number of runs to train",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="number of layers in the model"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="hidden dimension size",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=1024,
        help="batch size",
    )
    parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=[10, 10, 10],
        help="fanouts of each layer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay",
    )
    parser.add_argument("--var_try", type=int, default=5)

    parser.add_argument("--pfn", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=-1)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--end_at", type=int, default=3)
    parser.add_argument("--queue_size", type=int, default=-1)
    parser.add_argument("--grad_normlized_add", action="store_true")
    parser.add_argument("--grad_normlized_add_alpha", type=float, default=0.9)

    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(name=args.dataset))
    g = dataset[0]

    # assert num_nodes is same as the number of nodes in the graph
    print(f"[DEBUG] graph.number_of_nodes(): {g.number_of_nodes()}")
    if args.num_nodes == -1:
        args.num_nodes = g.number_of_nodes()
    else:
        assert args.num_nodes == g.number_of_nodes(), "args.num_nodes is not the same as the number of nodes in the graph"
    print(f"[DEBUG] args.num_nodes: {args.num_nodes}")

    if args.dataset == "ogbn-arxiv":
        # add reverse edges
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)

        # add self-loop
        print(f"Total edges before adding self-loop {g.num_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.num_edges()}")
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model_config = [in_size, args.hidden_dim, out_size]


    # # model training
    # print("Training...")
    # train(args, device, g, dataset, model_config, num_classes)

    # test dataloader for logging gradients
    # print("Testing dataloader for logging gradients...")
    # test_dataloader_for_logging_gradients(args, device, g, dataset, model_config, num_classes)

    # train with logging gradients
    # print("Training with logging gradients...")
    # train_with_logging_gradients(args, device, g, dataset, model_config, num_classes)

    # train in K-neighbor sampling try
    # print("Training with K-neighbor sampling try...")
    # train_with_K_neighbor_sampling_try(args, device, g, dataset, model_config, num_classes)

    # train with exact node embedding
    # print("Training with exact node embedding...")
    # train_with_exact_node_embedding(args, device, g, dataset, model_config, num_classes)

    # train with exact node embedding and variance reduction
    # print("Training with exact node embedding and variance reduction...")
    # train_with_exact_embedding_and_variance_reduction(args, device, g, dataset, model_config, num_classes)

    # train with logging embedding gradients
    # logging the variance of embedding. it means the inaccuracy of message.
    # the variance's change with the number of neighbors
    # the variance's change with the gradient norm on embedding
    # 
    print("Training with logging embedding gradients...")
    train_with_logging_embedding_gradients(args, device, g, dataset, model_config, num_classes)

    # test the model
    # print("Testing...")
    # acc = layerwise_infer(
    #     device, g, dataset.test_idx, model, num_classes, batch_size=100000
    # )
    # print("Test Accuracy {:.4f}".format(acc.item()))

