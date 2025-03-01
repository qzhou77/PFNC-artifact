import time
import hydra
from omegaconf import OmegaConf
import os

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, dropout)
from torch_geometric_autoscale.data import get_ppi

torch.manual_seed(123)

def full_train(model, data, criterion, optimizer, max_steps, grad_norm=None,
                edge_dropout=0.0, run=0, epoch=0, conf=None):
    model.train()

    total_loss = total_examples = 0
    batch_id = 0
    x = data.x.to(model.device)
    adj_t = data.adj_t.to(model.device)
    y = data.y.to(model.device)
    train_mask = data.train_mask.to(model.device)

    if train_mask.sum() == 0:
        return 0
    
    # We make use of edge dropout on ogbn-products to avoid overfitting.
    adj_t = dropout(adj_t, p=edge_dropout)

    if isinstance(optimizer, list):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
    else:
        optimizer.zero_grad()
    out = model(x, adj_t)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    if grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    if isinstance(optimizer, list):
        optimizer[0].step()
    if model.pfn is not None:
        model.pfn.detached_grad_correction()
    if hasattr(conf, 'save_grad_epoch') and epoch in conf.save_grad_epoch:
        all_grad_dict = {"train_mask": train_mask.cpu()}
        for name, param in model.named_parameters():
            # print(name, type(param.grad))
            all_grad_dict[name] = param.grad
        torch.save(all_grad_dict, conf.save_grad_path.replace(".pth", f"_run{run}_epoch{epoch}_batch{batch_id}.pth"))
        batch_id += 1
    if 'no_step' in os.environ:
        optimizer.zero_grad()
        # print(f"[DEBUG] no_step")
    else:
        if isinstance(optimizer, list):
            optimizer[1].step()
        else:
            optimizer.step()

    total_loss += float(loss) * int(train_mask.sum())
    total_examples += int(train_mask.sum())

    return total_loss / total_examples


def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None,
               edge_dropout=0.0, run=0, epoch=0, conf=None):
    model.train()

    total_loss = total_examples = 0
    batch_id = 0
    for i, (batch, batch_size, *args) in enumerate(loader):
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)

        if train_mask.sum() == 0:
            continue

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        adj_t = dropout(adj_t, p=edge_dropout)

        if isinstance(optimizer, list):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
        else:
            optimizer.zero_grad()
        out = model(x, adj_t, batch_size, *args)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        if isinstance(optimizer, list):
            optimizer[0].step()
        if model.pfn is not None:
            model.pfn.detached_grad_correction([args[0][:batch_size] for _ in range(model.num_layers)])
        if hasattr(conf, 'save_grad_epoch') and epoch in conf.save_grad_epoch:
            all_grad_dict = {"train_mask": train_mask.cpu()}
            for name, param in model.named_parameters():
                # print(name, type(param.grad))
                all_grad_dict[name] = param.grad
            torch.save(all_grad_dict, conf.save_grad_path.replace(".pth", f"_run{run}_epoch{epoch}_batch{batch_id}.pth"))
            batch_id += 1
        if 'no_step' in os.environ:
            optimizer.zero_grad()
            # print(f"[DEBUG] no_step")
        else:
            if isinstance(optimizer, list):
                optimizer[1].step()
            else:
                optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return total_loss / total_examples


@torch.no_grad()
def full_test(model, data, return_loss=False, criterion=None):
    model.eval()
    pred = model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()
    if return_loss:
        return pred, float(criterion(pred[data.train_mask], data.y[data.train_mask]))
    else:
        return pred


@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    try:
        edge_dropout = params.edge_dropout
    except:  # noqa
        edge_dropout = 0.0
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    t = time.perf_counter()
    print('Loading data...', end=' ', flush=True)
    data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    if conf.model.loop:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    if conf.model.norm:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)
    print(f"[DEBUG] len(train_loader): {len(train_loader)}")

    eval_loader = EvalSubgraphLoader(data, ptr,
                                     batch_size=params['batch_size'])

    if conf.dataset.name == 'ppi':
        val_data, _, _ = get_ppi(conf.root, split='val')
        test_data, _, _ = get_ppi(conf.root, split='test')
        if conf.model.loop:
            val_data.adj_t = val_data.adj_t.set_diag()
            test_data.adj_t = test_data.adj_t.set_diag()
        if conf.model.norm:
            val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=False)
            test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=False)

    t = time.perf_counter()
    print('Calculating buffer size...', end=' ', flush=True)
    # We reserve a much larger buffer size than what is actually needed for
    # training in order to perform efficient history accesses during inference.
    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader])
    if conf.dataset.name == 'reddit' and params.batch_size == 25:
        buffer_size = 2 * buffer_size
    print(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = data.adj_t.storage.rowcount()

    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size=buffer_size,
        pfn = conf.pfn,
        **params.architecture,
        **kwargs,
    ).to(device)

    if conf.pfn and conf.pfn_switch_epoch>0:
        assert model.pfn.grad_normlized_add==False and model.pfn.grad_add==False, "[Warning] If use pfn_switch_epoch, then shouldn't set grad_normlized_add or grad_add"
        

    results = torch.empty(params.runs)
    eclipse_times = torch.empty(params.runs)
    eclipse_times_no_history_fill = torch.empty(params.runs)
    # newly added time logger
    # 
    pre_time_before_train_all_runs = torch.empty(params.runs)
    all_epoch_time_with_testing = torch.empty(params.runs, params.epochs)
    all_epoch_time_without_testing = torch.empty(params.runs, params.epochs)
    # 
    for run in range(params.runs):
        model.reset_parameters()
        time_run_start = time.perf_counter()
        optimizer = torch.optim.Adam([
            dict(params=model.reg_modules.parameters(),
                weight_decay=params.reg_weight_decay),
            dict(params=model.nonreg_modules.parameters(),
                weight_decay=params.nonreg_weight_decay)
        ], lr=params.lr)

        t = time.perf_counter()
        print('Fill history...', end=' ', flush=True)
        mini_test(model, eval_loader)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        time_train_start = time.perf_counter()
        best_val_acc = test_acc = 0
        # 
        pre_time_before_train_all_runs[run] = time_train_start - time_run_start
        # print pre time
        print(f"[TIME], run = {run + 1:2d}, pre_time = {time_train_start - time_run_start:.2f}")
        # 
        for epoch in range(1, params.epochs + 1):
            if hasattr(conf, 'save_checkpoint_epoch') and epoch in conf.save_checkpoint_epoch:
                torch.save(model.state_dict(), conf.save_checkpoint_path.replace(".pth", f"_run{run}_epoch{epoch}.pth"))
            if hasattr(conf, 'load_checkpoint_epoch') and conf.load_checkpoint_epoch == epoch:
                model.load_state_dict(torch.load(conf.load_checkpoint_path))

            # 
            time_epoch_train_start = time.perf_counter()
            # 
            if conf.pfn:
                if epoch > conf.pfn_switch_epoch:
                    model.pfn.grad_normlized_add = True
                    model.pfn.grad_add = False
                else:
                    model.pfn.grad_normlized_add = False
                    model.pfn.grad_add = True

            if hasattr(conf, 'full_train') and conf.full_train:
                loss = full_train(model, data, criterion, optimizer, params.max_steps,
                                grad_norm, edge_dropout, run, epoch, conf)
            else:
                loss = mini_train(model, train_loader, criterion, optimizer,
                                params.max_steps, grad_norm, edge_dropout, run, epoch, conf)
            # 
            time_epoch_train_end = time.perf_counter()
            all_epoch_time_without_testing[run][epoch-1] = time_epoch_train_end - time_epoch_train_start
            # 
            if conf.full_test:
                out, loss = full_test(model, data, conf.full_loss, criterion)
            else:
                out = mini_test(model, eval_loader)
            train_acc = compute_micro_f1(out, data.y, data.train_mask)

            if conf.dataset.name != 'ppi':
                val_acc = compute_micro_f1(out, data.y, data.val_mask)
                tmp_test_acc = compute_micro_f1(out, data.y, data.test_mask)
            else:
                # We need to perform inference on a different graph as PPI is an
                # inductive dataset.
                val_acc = compute_micro_f1(full_test(model, val_data), val_data.y)
                tmp_test_acc = compute_micro_f1(full_test(model, test_data),
                                                test_data.y)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                results[run] = test_acc
            if epoch % conf.log_every == 0:
                # print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, '
                #       f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                #       f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')
                print(f"[INFO], run = {run + 1:2d}, epoch = {epoch:3d}, loss = {loss:.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {tmp_test_acc:.4f}, best_val = {best_val_acc:.4f}, bestval_test = {test_acc:.4f}")

            #
            time_epoch_train_with_testing_end = time.perf_counter()
            all_epoch_time_with_testing[run][epoch-1] = time_epoch_train_with_testing_end - time_epoch_train_start
            # print moving average epoch time
            total_epoch_num_till_now = run * params.epochs + epoch
            print(f"[TIME], run = {run + 1:2d}, epoch = {epoch:3d},\n", 
                  f"[TIME]--this epoch time w./w.o. testing = {time_epoch_train_with_testing_end - time_epoch_train_start:.4f},{time_epoch_train_end - time_epoch_train_start:.4f}\n" ,
                  f"[TIME]--avg(this run) epoch time w./w.o. testing = [{all_epoch_time_with_testing[run][:epoch].mean():.4f},{all_epoch_time_without_testing[run][:epoch].mean():.4f}],\n",
                  f"[TIME]--avg(all epoch across runs) epoch time w./w.o. testing = [{all_epoch_time_with_testing.view(-1)[:total_epoch_num_till_now].mean():.4f},{all_epoch_time_without_testing.view(-1)[:total_epoch_num_till_now].mean():.4f}],")
            # 
        time_run_end = time.perf_counter()
        eclipse_times[run] = time_run_end - time_run_start
        eclipse_times_no_history_fill[run] = time_run_end - time_train_start


    print('=========================')
    print(f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')
    print(f'Average time: {eclipse_times.mean():.2f} ± {eclipse_times.std():.2f} seconds')
    print(f'Average time without history fill: {eclipse_times_no_history_fill.mean():.2f} ± {eclipse_times_no_history_fill.std():.2f} seconds')


if __name__ == "__main__":
    # print current local time
    print("Current local time: ", time.asctime(time.localtime(time.time())))
    main()
