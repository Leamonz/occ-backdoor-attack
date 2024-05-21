from collections import defaultdict
import math
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from sklearn.metrics import f1_score, accuracy_score


class MetricMonitor:
    # metric is a dict. {metric_name : metric}
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.metric = defaultdict(lambda: {"val": 0, "count": 0, "avg": .0})

    def update(self, metric_name, val):
        metric = self.metric[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __srt__(self):
        return " | ".join(
            "{metric_name}: {avg:.{float_precision}f}".format(
                metric_name=metric_name, avg=metric["avg"],
                float_precision=self.float_precision
            )
            for (metric_name, metric) in self.metric.items()
        )


def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support %s' % lr_schedule_type)
    return lr


def adjust_learning_rate(optimizer, epoch, params: dict, batch=0, nBatch=None):
    new_lr = calc_learning_rate(epoch, params['lr'], params['num_epoch'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def get_device(device):
    if device == "default":
        ret = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        ret = device
    return ret


def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    return f1_score(target, y_pred, average='macro')


def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# return labels dict in csv file.
# key means class, value means sequence, e.g. 0,1,...
# >>> type(labels)
# dict
def get_label_and_dict(csv_path, mode, dtype=str):
    df = pd.read_csv(csv_path, dtype=dtype)
    le = LabelEncoder()
    if mode == 'train':
        le.fit(df['label'])
        df['label'] = le.transform((df['label']))
    if mode == 'infer':
        le.fit(df['label'])
    return dict(zip(le.classes_, le.transform(le.classes_))), df


# return invert dict
def get_invert_dict(label_map):
    return {v: k for k, v in label_map.items()}


def infer_overall_acc(infer_csv_path, dtype=str):
    df = pd.read_csv(infer_csv_path, dtype=dtype)
    all_image = len(df['label'])
    count = 0
    for ground_truth_label, result in zip(df['label'].items(), df['result'].items()):
        if ground_truth_label == result:
            count += 1
    acc = count / all_image
    return acc


def infer_clean_acc(infer_csv_path, normal, dtype=str):
    df = pd.read_csv(infer_csv_path, dtype=dtype)
    all_image = 0
    count = 0
    for index, row in df.iterrows():
        if row['image'].find(normal) != -1:
            all_image += 1
            if row['label'] == row['result']:
                count += 1
    acc = count / all_image
    return acc


def infer_poisoned_acc(infer_csv_path, trigger, dtype=str):
    df = pd.read_csv(infer_csv_path, dtype=dtype)
    all_image = 0
    count = 0
    for index, row in df.iterrows():
        if row['image'].find(trigger) != -1:
            all_image += 1
            if row['label'] == row['result']:
                count += 1
    acc = count / all_image
    return acc


def infer_poisoned_acc_all2all(infer_csv_path, trigger, dtype=str):
    df = pd.read_csv(infer_csv_path, dtype=dtype)
    all_image = 0
    count = 0
    for index, row in df.iterrows():
        if row['image'].find(trigger) != -1:
            all_image += 1
            target_label = int(row['ground_truth_label']) % 10 + 1
            target_label = str(target_label).zfill(3)
            if row['result'] == target_label:
                count += 1
    acc = count / all_image
    return acc


def plot_curve(csv_dir):
    csv_files = os.listdir(csv_dir)
    first_csv = pd.read_csv(csv_files[0])
    data = first_csv.to_numpy()
    for each_csv in csv_files[1:]:
        data_a = pd.read_csv(each_csv).to_numpy()
        data = np.vstack((data, data_a))
    df = pd.DataFrame(data, columns=["epoch", "train_loss", "val_loss"])
    sns.relplot(x="epoch", y='train_loss', data=df)
    sns.relplot(x="epoch", y="val_loss", data=df)
    plt.legend(loc="upper left", labels=["train loss", "validation loss"])
    plt.show()


if __name__ == "__main__":
    plot_curve(r"./loss")
    