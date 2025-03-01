import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
import time
import os

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute, models,
                                       SubgraphLoader, compute_micro_f1)

torch.manual_seed(123)
criterion = torch.nn.CrossEntropyLoss()


def train(run, epoch, model, loader, optimizer, grad_norm=None, conf=None):
    model.train()

    total_loss = total_examples = 0
    batch_id = 0
    for batch, batch_size, n_id, _, _ in loader:
        batch = batch.to(model.device)
        n_id = n_id.to(model.device)

        mask = batch.train_mask[:batch_size]
        mask = mask[:, run] if mask.dim() == 2 else mask
        if mask.sum() == 0:
            continue

        if isinstance(optimizer, list):
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
        else:
            optimizer.zero_grad()
        out = model(batch.x, batch.adj_t, batch_size, n_id)
        loss = criterion(out[mask], batch.y[:batch_size][mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        if isinstance(optimizer, list):
            optimizer[0].step()
        if model.pfn is not None:
            model.pfn.detached_grad_correction([n_id[:batch_size] for _ in range(model.num_layers)])
        if hasattr(conf, 'save_grad_epoch') and epoch in conf.save_grad_epoch:
            all_grad_dict = {"nid": n_id.cpu()}
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
        total_loss += float(loss) * int(mask.sum())
        total_examples += int(mask.sum())

    return total_loss / total_examples


@torch.no_grad()
def test(run, model, data):
    model.eval()

    train_mask = data.train_mask
    train_mask = train_mask[:, run] if train_mask.dim() == 2 else train_mask

    val_mask = data.val_mask
    val_mask = val_mask[:, run] if val_mask.dim() == 2 else val_mask

    test_mask = data.test_mask
    test_mask = test_mask[:, run] if test_mask.dim() == 2 else test_mask

    out = model(data.x, data.adj_t)
    train_acc = compute_micro_f1(out, data.y, train_mask)
    val_acc = compute_micro_f1(out, data.y, val_mask)
    test_acc = compute_micro_f1(out, data.y, test_mask)

    return train_acc, val_acc, test_acc


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    model_name, dataset_name = conf.model.name, conf.dataset.name
    conf.model.params = conf.model.params[dataset_name]
    params = conf.model.params
    print(OmegaConf.to_yaml(conf))
    if isinstance(params.grad_norm, str):
        params.grad_norm = None

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    data, in_channels, out_channels = get_data(conf.root, dataset_name)
    if conf.model.norm:
        data.adj_t = gcn_norm(data.adj_t)
    elif conf.model.loop:
        data.adj_t = data.adj_t.set_diag()

    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                            shuffle=True, num_workers=params.num_workers,
                            persistent_workers=params.num_workers > 0)
    print(f"[DEBUG] len(loader): {len(loader)}")

    data = data.clone().to(device)  # Let's just store all data on GPU...

    GNN = getattr(models, model_name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,  # ... and put histories on GPU as well.
        pfn = conf.pfn,
        **params.architecture,
    ).to(device)

    results = torch.empty(params.runs)
    # pbar = tqdm(total=params.runs * params.epochs)
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

        test(0, model, data)  # Fill history.

        best_val_acc = 0

        time_train_start = time.perf_counter()
        # 
        pre_time_before_train_all_runs[run] = time_train_start - time_run_start
        # print pre time
        # print(f"[TIME], run = {run + 1:2d}, pre_time = {time_train_start - time_run_start:.2f}")
        # 
        for epoch in range(params.epochs):
            if hasattr(conf, 'save_checkpoint_epoch') and epoch in conf.save_checkpoint_epoch:
                torch.save(model.state_dict(), conf.save_checkpoint_path.replace(".pth", f"_run{run}_epoch{epoch}.pth"))
            if hasattr(conf, 'load_checkpoint_epoch') and epoch in conf.load_checkpoint_epoch:
                model.load_state_dict(torch.load(conf.load_checkpoint_path.replace(".pth", f"_run{run}_epoch{epoch}.pth")))

            # 
            time_epoch_train_start = time.perf_counter()
            # 
            loss = train(run, epoch, model, loader, optimizer, params.grad_norm, conf)
            # 
            time_epoch_train_end = time.perf_counter()
            all_epoch_time_without_testing[run][epoch-1] = time_epoch_train_end - time_epoch_train_start
            # 
            train_acc, val_acc, test_acc = test(run, model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                results[run] = test_acc
            print(f"[INFO], run = {run + 1:2d}, epoch = {epoch + 1:3d}, loss = {loss:.5f}, train = {train_acc:.4f}, val = {val_acc:.4f}, test = {test_acc:.4f}, best_val = {best_val_acc:.4f}, bestval_test = {results[run]:.4f}")
            # pbar.set_description(f'Mini Acc: {100 * results[run]:.2f}')
            # pbar.update(1)
            
            #
            time_epoch_train_with_testing_end = time.perf_counter()
            all_epoch_time_with_testing[run][epoch-1] = time_epoch_train_with_testing_end - time_epoch_train_start
            # print moving average epoch time
            total_epoch_num_till_now = run * params.epochs + epoch
            # print(f"[TIME], run = {run + 1:2d}, epoch = {epoch:3d},\n", 
            #       f"[TIME]--this epoch time w./w.o. testing = {time_epoch_train_with_testing_end - time_epoch_train_start:.4f},{time_epoch_train_end - time_epoch_train_start:.4f}\n" ,
            #       f"[TIME]--avg(this run) epoch time w./w.o. testing = [{all_epoch_time_with_testing[run][:epoch].mean():.4f},{all_epoch_time_without_testing[run][:epoch].mean():.4f}],\n",
            #       f"[TIME]--avg(all epoch across runs) epoch time w./w.o. testing = [{all_epoch_time_with_testing.view(-1)[:total_epoch_num_till_now].mean():.4f},{all_epoch_time_without_testing.view(-1)[:total_epoch_num_till_now].mean():.4f}],")
            # 
    # pbar.close()
    print(f'Mini Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')


if __name__ == "__main__":
    main()
