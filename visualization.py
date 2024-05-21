import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import os
import glob
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
# https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/49884009#49884009
ALL_EPOCH = 30
K_FOLD = 3
NUM_SINGLE_ATTACK = 90
FONTSIZE = 24


def calculate_std_and_mean_by_each_epoch(csv_dir: str, which_epoch: int):
    # all_train_loss = np.zeros(shape=(1, K_FOLD * NUM_SINGLE_ATTACK))
    # all_val_loss = np.zeros(shape=(1, K_FOLD * NUM_SINGLE_ATTACK))
    all_train_loss_in_this_epoch = []
    all_val_loss_in_this_epoch = []
    for each_csv in glob.glob(os.path.join(csv_dir + r"\\*.csv")):
        # each_np = np.loadtxt(each_csv, delimiter=",")
        each_df = pd.read_csv(each_csv, delimiter=",")
        each_train_loss = each_df.iloc[which_epoch::ALL_EPOCH, 1].tolist()
        each_val_loss = each_df.iloc[which_epoch::ALL_EPOCH, 2].tolist()
        all_train_loss_in_this_epoch.extend(each_train_loss)
        all_val_loss_in_this_epoch.extend(each_val_loss)
    return all_train_loss_in_this_epoch, all_val_loss_in_this_epoch


def draw_loss_with_standard_dviation(x, mean: list[list], std: list[list], lr, labels, name='loss'):
    # https://blog.csdn.net/monotonomo/article/details/83342768
    # sns.set_style('whitegrid')
    color = cm.viridis(0.5)
    # color = cm.plasma(0.5)
    f, ax = plt.subplots(1, 1)
    colors = [cm.RdPu(0.8), cm.PuBu(0.8)]
    for idx in range(len(mean)):
        ax.plot(x, mean[idx], color=colors[idx], label=labels[idx])
        r1 = list(map(lambda x: x[0] - x[1], zip(mean[idx], std[idx])))
        r2 = list(map(lambda x: x[0] + x[1], zip(mean[idx], std[idx])))
        ax.fill_between(x, r1, r2, color=colors[idx], alpha=0.2)
    # ax.plot(clean_loss, mean, color=colors[1], label="Clean Loss")
    ax.legend()
    ax.set_title(label=f"LR={lr}")
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # let boundary close
    ax.set(xlim=(x[0], x[-1]))
    exp_dir = 'Plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    f.savefig(os.path.join('Plot', name + '.svg'), dpi=1000)


def mesh(csv_path, CA_name, ASR_name):
    # data parse
    cm = plt.colormaps["plasma"]
    # https://www.analyticsvidhya.com/blog/2020/09/colormaps-matplotlib/
    plt.set_cmap(cm)
    df = pd.read_csv(csv_path, index_col=0)
    # overall_acc_np = np.ndarray(shape=(10, 10), dtype=float)
    clean_acc_np = np.ndarray(shape=(10, 10), dtype=float)
    attack_successful_rate_np = np.ndarray(shape=(10, 10), dtype=float)
    # row: clean col:target
    for col in range(0, 10):
        for row in range(0, 10):
            if row == col:
                # overall_acc_np[row, col] = 0
                clean_acc_np[row, col] = 0
                attack_successful_rate_np[row, col] = 0
                continue
            clean_label = row + 1
            target_label = col + 1
            # [str(row).zfill(3) + "_" + str(col).zfill(3)]: index foramt in dataframe
            # overall_acc_np[row, col] = df['overall_acc'][str(clean_label).zfill(3) + "_" + str(target_label).zfill(3)]
            clean_acc_np[row, col] = df['CA'][str(clean_label).zfill(3) + "_" + str(target_label).zfill(3)]
            attack_successful_rate_np[row, col] = df['ASR'][
                str(clean_label).zfill(3) + "_" + str(target_label).zfill(3)]
    # visualization
    # plt.style.use('_mpl-gallery-nogrid')
    # CA
    fig, ax = plt.subplots(1, 1)
    # fig size
    X = np.arange(-0.5, 10, 1)
    Y = np.arange(-0.5, 10, 1)
    pcm_0 = ax.pcolormesh(X, Y, clean_acc_np)
    # pcm = ax.pcolormesh(clean_acc_np)
    # ax.set(xlim=(-0.5, 9.5))
    # x & y coordinate ticks
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))
    fig.colorbar(pcm_0, ax=ax)
    # ax.set_xlabel('ground truth label', fontsize=FONTSIZE)
    # ax.set_ylabel('target label', fontsize=FONTSIZE)
    # ax.set_title(label="CA(%)", fontsize=FONTSIZE)
    exp_dir = 'Plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    fig.savefig(os.path.join('Plot', CA_name), dpi=1000, bbox_inches='tight')
    # SAR
    fig, ax = plt.subplots(1, 1)
    pcm_1 = ax.pcolormesh(X, Y, attack_successful_rate_np)
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))
    fig.colorbar(pcm_1, ax=ax)
    # ax.set_xlabel('ground truth label', fontsize=FONTSIZE)
    # ax.set_ylabel('target label', fontsize=FONTSIZE)
    # ax.set_title(label="ASR(%)", fontsize=FONTSIZE)
    fig.savefig(os.path.join('Plot', ASR_name), dpi=1000, bbox_inches='tight')


def calculate_each_case_CA_ASR(csv_dir, num_poison):
    # print(num_poison)
    all_CA_for_each_rate = []
    all_ASR_for_each_rate = []
    # print(glob.glob(r"./result/injection_rate/*/analysis.csv"))
    for each_csv in glob.glob(os.path.join(csv_dir, r"*/analysis.csv")):
        # print(each_csv)
        each_df = pd.read_csv(each_csv)
        each_CA = each_df.iloc[num_poison // 5, 1]
        each_ASR = each_df.iloc[num_poison // 5, 2]
        all_CA_for_each_rate.append(each_CA)
        all_ASR_for_each_rate.append(each_ASR)
    return all_CA_for_each_rate, all_ASR_for_each_rate


def draw_CA_ASR_with_standard_dviation(x, mean: list[list], std: list[list], labels, name='loss'):
    # https://blog.csdn.net/monotonomo/article/details/83342768
    # sns.set_style('whitegrid')
    # color = cm.viridis(0.5)
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 1, 1)
    colors = [cm.RdPu(0.8), cm.PuBu(0.8)]
    for idx in range(len(mean)):
        ax.plot(x, mean[idx], color=colors[idx], label=labels[idx])
        r1 = list(map(lambda x: x[0] - x[1], zip(mean[idx], std[idx])))
        r2 = list(map(lambda x: x[0] + x[1], zip(mean[idx], std[idx])))
        ax.fill_between(x, r1, r2, color=colors[idx], alpha=0.2)
    # ax.plot(clean_loss, mean, color=colors[1], label="Clean Loss")
    # ax.legend(loc="lower right")
    # ax.set_title(label=f"LR={lr}")
    # ax.set_xlabel('Number of Poisoned Images')
    # ax.set_ylabel('Accuracy')
    # let boundary close
    ax.set(xlim=(x[0], x[-1]), ylim=(0, 1.1))
    exp_dir = 'Plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    plt.savefig(os.path.join('Plot', name + '.svg'), dpi=1000)


if __name__ == "__main__":
    '''
    cvs_dir = r"loss/csv/resnet50_result/normal_5k/train"
    all_train_loss_list = []
    all_val_loss_list = []
    for each_epoch in range(ALL_EPOCH):
        all_train_loss_in_this_epoch, all_val_loss_in_this_epoch = calculate_std_and_mean_by_each_epoch(cvs_dir,
                                                                                                        each_epoch)
        all_train_loss_list.append(all_train_loss_in_this_epoch)
        all_val_loss_list.append(all_val_loss_in_this_epoch)
    all_train_loss_mean = np.average(all_train_loss_list, axis=1).reshape(-1)
    all_val_loss_mean = np.average(all_val_loss_list, axis=1).reshape(-1)
    all_train_loss_std = np.std(all_train_loss_list, axis=1).reshape(-1)
    all_val_loss_std = np.std(all_val_loss_list, axis=1).reshape(-1)
    x = np.linspace(0, ALL_EPOCH, ALL_EPOCH)
    draw_loss_with_standard_dviation(x, all_train_loss_mean, all_train_loss_std)
    '''
    '''
    shufflenet_csv_path_1 = r"result/shufflenet/normal_4.5k/analysis.csv"
    shufflenet_csv_path_2 = r"result/shufflenet/4k_4.5k/analysis.csv"
    mesh(shufflenet_csv_path_1, "CA1.png", "ASR1.png")
    mesh(shufflenet_csv_path_2, "CA2.png", "ASR2.png")
    '''
'''
if __name__ == "__main__":
    ### draw inject rate accuracy figures
    all_CA_list = []
    all_ASR_list = []
    for num_poison in range(0, 36, 5):
        all_CA_for_this_rate, all_ASR_for_this_rate = calculate_each_case_CA_ASR("./result/injection_rate/",
                                                                                 num_poison)
        all_CA_list.append(all_CA_for_this_rate)
        all_ASR_list.append(all_ASR_for_this_rate)
    all_CA_list = np.asarray(all_CA_list)
    all_ASR_list = np.asarray(all_ASR_list)

    all_CA_mean = np.average(all_CA_list, axis=1).reshape(-1)
    all_ASR_mean = np.average(all_ASR_list, axis=1).reshape(-1)
    all_CA_std = np.std(all_CA_list, axis=1).reshape(-1)
    all_ASR_std = np.std(all_ASR_list, axis=1).reshape(-1)
    x = np.linspace(0, 35, len(all_CA_mean))
    # print(all_ASR_mean)
    # print(all_ASR_std)
    draw_CA_ASR_with_standard_dviation(
        x,
        [all_CA_mean, all_ASR_mean],
        [all_CA_std, all_ASR_std],
        labels=["CA", "ASR"],
        name="Accuracy"
    )


    ### draw loss figures
    # plt.figure(figsize=(21, 3))
    # title = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    # for idx, lr in enumerate(["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]):
    #     # lr = "1e-5"
    #     csv_dir = rf"D:\Program_work\occ-face-recognize\loss\resnet50_{lr}\loss\csv\resnet50\normal_5k\train"
    #     csv_clean_dir = rf"D:\Program_work\occ-face-recognize\loss\resnet_clean_{lr}"
    #     all_train_loss_list = []
    #     all_val_loss_list = []
    #     for each_epoch in range(ALL_EPOCH):
    #         all_train_loss_in_this_epoch, all_val_loss_in_this_epoch = calculate_std_and_mean_by_each_epoch(csv_dir,
    #                                                                                                         each_epoch)
    #         all_train_loss_list.append(all_train_loss_in_this_epoch)
    #         all_val_loss_list.append(all_val_loss_in_this_epoch)
    #
    #     all_train_loss_list = np.asarray(all_train_loss_list)
    #     all_val_loss_list = np.asarray(all_val_loss_list)
    #
    #     all_train_loss_mean = np.average(all_train_loss_list, axis=1).reshape(-1)
    #     all_val_loss_mean = np.average(all_val_loss_list, axis=1).reshape(-1)
    #     all_train_loss_std = np.std(all_train_loss_list, axis=1).reshape(-1)
    #     all_val_loss_std = np.std(all_val_loss_list, axis=1).reshape(-1)
    #
    #     clean_all_train_loss_list = []
    #     clean_all_val_loss_list = []
    #     for each_epoch in range(ALL_EPOCH):
    #         clean_all_train_loss_in_this_epoch, clean_all_val_loss_in_this_epoch = calculate_std_and_mean_by_each_epoch(
    #             csv_clean_dir,
    #             each_epoch)
    #         clean_all_train_loss_list.append(clean_all_train_loss_in_this_epoch)
    #         clean_all_val_loss_list.append(clean_all_val_loss_in_this_epoch)
    #
    #     clean_all_train_loss_list = np.asarray(clean_all_train_loss_list)
    #     clean_all_val_loss_list = np.asarray(clean_all_val_loss_list)
    #
    #     clean_all_train_loss_mean = np.average(clean_all_train_loss_list, axis=1).reshape(-1)
    #     clean_all_val_loss_mean = np.average(clean_all_val_loss_list, axis=1).reshape(-1)
    #     clean_all_train_loss_std = np.std(clean_all_train_loss_list, axis=1).reshape(-1)
    #     clean_all_val_loss_std = np.std(clean_all_val_loss_list, axis=1).reshape(-1)
    #
    #     x = np.linspace(0, ALL_EPOCH, ALL_EPOCH)
    #     mean = [all_train_loss_mean, clean_all_train_loss_mean]
    #     std = [all_train_loss_std, clean_all_train_loss_std]
    #     labels = ["Attack Loss", "Clean Loss"]
    #     ax = plt.subplot(1, 5, idx + 1)
    #     # f, ax = plt.subplots(1, 1)
    #     colors = [cm.RdPu(0.8), cm.PuBu(0.8)]
    #     for idx in range(len(mean)):
    #         ax.plot(x, mean[idx], color=colors[idx], label=labels[idx])
    #         r1 = list(map(lambda x: x[0] - x[1], zip(mean[idx], std[idx])))
    #         r2 = list(map(lambda x: x[0] + x[1], zip(mean[idx], std[idx])))
    #         ax.fill_between(x, r1, r2, color=colors[idx], alpha=0.2)
    #     # ax.legend()
    #     # ax.set_title(label=f"LR={lr}")
    #     # let boundary close
    #     ax.set(xlim=(x[0], x[-1]))
    # plt.savefig("Plot/loss.svg", dpi=1200)
    # plt.show()
    # draw_loss_with_standard_dviation(x,
    #                                  [all_train_loss_mean, clean_all_train_loss_mean],
    #                                  [all_train_loss_std, clean_all_train_loss_std],
    #                                  labels=["Attack Loss", "Clean Loss"],
    #                                  lr=lr,
    #                                  name=f'loss_{lr}')
    # mesh()
'''
