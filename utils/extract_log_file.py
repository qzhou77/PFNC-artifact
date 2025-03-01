import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def load_log_file(log_file, rt_val=False):
    """
    Load log file and extract relevant information.

    Parameters:
    log_file (str): Path to the log file.

    Returns:
    run_list (numpy.ndarray): Array of run numbers.
    epoch_list (numpy.ndarray): Array of epoch numbers.
    loss_list (numpy.ndarray): Array of loss values.
    train_list (numpy.ndarray): Array of train values.
    val_list (numpy.ndarray): Array of validation values.
    test_list (numpy.ndarray): Array of test values.
    """
    with open(log_file, "r") as f:
        lines = f.readlines()

    # get line begin with "Mini Acc"
    acc = [line for line in lines if line.startswith("Mini Acc")]
    # lines contain "\pm" or "±"
    acc_with_pm = [line for line in lines if "\pm" in line or "±" in line]
    try:
        if len(acc) > 0:
            print(f"[INFO] {acc[0]}, file: {log_file}")
        elif len(acc_with_pm) > 0:
            print(f"[INFO] {acc_with_pm[0]}, file: {log_file}")
    except:
        pass
    # Mini Acc: 79.77 ± 1.83
    # get value
    try:
        avg = float(acc[0].split("±")[0].split(":")[-1])
    except:
        avg = 0
    try:
        std = float(acc[0].split("±")[1])
    except:
        std = 0

    # remain line begin with [Run
    lines = [line for line in lines if line.startswith("[INFO],")]

    # item_name_list = ["run", "epoch", "loss", "train", "val", "test", "best_val", "bestval_test"]
    run_list, epoch_list, loss_list, train_list, val_list, test_list, best_val_list, bestval_test_list = [
    ], [], [], [], [], [], [], []

    for l in lines:
        # [INFO], run =  8, epoch =   2, loss = 1.93375, train = 0.2857, val = 0.3760, test = 0.3650, best_val = 0.3760, bestval_test = 0.3650
        assign_val_list = l.split(", ")[1:]
        d = {}
        for asn in assign_val_list:
            exec(asn, d)
        run_list.append(d['run'])
        epoch_list.append(d['epoch'])
        loss_list.append(d['loss'])
        train_list.append(d['train'])
        val_list.append(d['val'])
        test_list.append(d['test'])
        best_val_list.append(d['best_val'])
        bestval_test_list.append(d['bestval_test'])

    run_list = np.array(run_list)
    epoch_list = np.array(epoch_list)
    loss_list = np.array(loss_list)
    train_list = np.array(train_list)
    val_list = np.array(val_list)
    test_list = np.array(test_list)
    best_val_list = np.array(best_val_list)
    bestval_test_list = np.array(bestval_test_list)

    # check if epoch in every run is ordered
    run_num = run_list.max()
    for i in range(run_num):
        check_epoch = epoch_list[run_list == i + 1]
        assert np.all(np.diff(check_epoch) ==
                      1), f"run {i+1} epoch is not ordered"

    print(f"Finish loading log file {log_file}")
    print(f"{log_file.split('/')[-1]}")
    print(
        f"Total run number: {run_num}, epoch number per run: {epoch_list.max()}")

    if rt_val:
        return (avg, std), run_list, epoch_list, loss_list, train_list, val_list, test_list, best_val_list, bestval_test_list
    else:
        return run_list, epoch_list, loss_list, train_list, val_list, test_list, best_val_list, bestval_test_list


# set font size to very small
plt.rcParams.update({'font.size': 6})


def plot_avg_std_vs_epoch(epoch_list, val_list, val_name, run_num, wait_new_fig=False, line_type="-"):
    """
    Plot average and standard deviation of a value across epochs.

    Parameters:
    epoch_list (numpy.ndarray): Array of epoch numbers.
    val_list (numpy.ndarray): Array of values.
    val_name (str): Name of the value.
    run_num (int): Number of runs.
    wait_new_fig (bool, optional): Whether to wait for a new figure. Defaults to False.
    """
    # plot avg and std across runs
    list_of_val_per_run = []
    for i_run in range(run_num):
        list_of_val_per_run.append(val_list[run_list == i_run + 1])
    list_of_val_per_run = np.array(list_of_val_per_run)
    avg_val = np.mean(list_of_val_per_run, axis=0)
    std_val = np.std(list_of_val_per_run, axis=0)
    plot_epoch_list = np.arange(1, epoch_list.max() + 1)
    plt.plot(plot_epoch_list, avg_val,
             label=f"{val_name} avg", linestyle=line_type)
    plt.fill_between(plot_epoch_list, avg_val - std_val,
                     avg_val + std_val, alpha=0.1, label=f"{val_name} std")
    plt.xlabel("epoch")
    plt.ylabel(val_name)
    plt.legend()
    if not wait_new_fig:
        plt.show()


def plot_avg_std_vs_epoch_v1(epoch_list, run_list, val_list, val_name, run_num, wait_new_fig=False, 
                             line_type="-", linewidth=0.2, 
                             color="none", 
                             target_acc=-1, print_max_val=False, 
                             st_epoch=0, ed_epoch=-1, 
                             gaussian_smooth=False, gaussian_sigma=2, pre_smooth=False, 
                             specific_run=0, cutoff_epoch=0, 
                             ):
    """
    Plot average and standard deviation of a value across epochs.

    Parameters:
    epoch_list (numpy.ndarray): Array of epoch numbers.
    val_list (numpy.ndarray): Array of values.
    val_name (str): Name of the value.
    run_num (int): Number of runs.
    wait_new_fig (bool, optional): Whether to wait for a new figure. Defaults to False.
    """
    # plot avg and std across runs
    list_of_val_per_run = []
    for i_run in range(run_num):
        list_of_val_per_run.append(val_list[run_list == i_run + 1])
    list_of_val_per_run = np.array(list_of_val_per_run)
    if specific_run > 0:
        avg_val = list_of_val_per_run[specific_run - 1]
        std_val = np.zeros_like(avg_val)
    else:
        avg_val = np.mean(list_of_val_per_run, axis=0)
        std_val = np.std(list_of_val_per_run, axis=0)
    if cutoff_epoch != 0:
        avg_val = avg_val[:cutoff_epoch]
        std_val = std_val[:cutoff_epoch]

    if pre_smooth and gaussian_smooth:
        avg_val = gaussian_filter1d(avg_val, sigma=gaussian_sigma)
    if print_max_val:
        print(f"[INFO] beatval_test max: {avg_val.max()}")
    if color == "none":
        print(
            f"[INFO] on the fly  $mean \pm std$   : ${avg_val[-1]:.8f} \pm {std_val[-1]:.4f}$")
    elif color == "red":
        print(
            f"\033[91m[INFO RAW] on the fly  $mean \pm std$   : ${avg_val[-1]:.8f} \pm {std_val[-1]:.4f}$\033[0m")
    if target_acc > 0:
        # find the first epoch that avg_val >= target_acc
        try:
            idx = np.where(avg_val >= target_acc)[0][0]
            # print(f"[DEBUG] {idx}")
            print(
                f"\033[92m[INFO first reach epoch] Epoch {idx+1} avg_val >= {target_acc}\033[0m")
        except:
            print(
                f"\033[91m[INFO first reach epoch] Epoch not found, avg_val always < {target_acc}\033[0m")
    if (not pre_smooth) and gaussian_smooth:
        avg_val = gaussian_filter1d(avg_val, sigma=gaussian_sigma)
    plot_epoch_list = np.arange(1, epoch_list.max() + 1)
    if cutoff_epoch != 0:
        plot_epoch_list = plot_epoch_list[:cutoff_epoch]
    plt.plot(plot_epoch_list[st_epoch:ed_epoch], avg_val[st_epoch:ed_epoch],
             label=f"{val_name} avg", linestyle=line_type, linewidth=linewidth)
    if ed_epoch != -1:
        plt.fill_between(plot_epoch_list[st_epoch:ed_epoch], avg_val[st_epoch:ed_epoch] -
                         std_val[st_epoch:ed_epoch], avg_val[st_epoch:ed_epoch] + std_val[st_epoch:ed_epoch], alpha=0.1)
    else:
        plt.fill_between(plot_epoch_list[st_epoch:], avg_val[st_epoch:] -
                         std_val[st_epoch:], avg_val[st_epoch:] + std_val[st_epoch:], alpha=0.1)
    plt.xlabel("epoch")
    plt.ylabel(val_name)
    num1 = 1.05
    num2 = 0
    num3 = 1
    num4 = 0
    plt.legend(loc=(num1, num2), ncol=num3, mode=num4)
    if not wait_new_fig:
        plt.show()