import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.dataset import ClassificationDataset
from lib.utils import get_device, get_label_and_dict, get_invert_dict
from lib.model import ClassificationNet, train, validate
from lib.preprocess import get_train_transforms, get_valid_transforms
import json
from sklearn.model_selection import StratifiedKFold
import os
import sys
import numpy as np
import pandas as pd

json_file = sys.argv[1]
# trigger frequency
trigger = sys.argv[2]
print(json_file)
with open(json_file, "r") as f:
    params = json.load(f)

# label
label_map, df = get_label_and_dict(params["train_dataset_csv"], mode='train', dtype=str)
label_inv_map = get_invert_dict(label_map)

clean_indice = []
poison_indice = []
for idx, row in df.iterrows():
    if trigger in row['image']:
        poison_indice.append(idx)
    else:
        clean_indice.append(idx)

df_clean = df.iloc[clean_indice]
df_poison = df.iloc[poison_indice]

# base config
device = get_device(params["device"])
best_acc = 0.0
model_name = params["model"]
num_class = params["num_class"]
num_epochs = params["num_epoch"]
image_size = params["image_size"]
batch_size = params["batch_size"]
num_worker = params["num_worker"]
lr = params["lr"]
weight_decay = params["weight_decay"]
model_save_dir = params["model_save_dir"]
n_split = params["n_split"]
loss_dir = "./loss"
all_train_losses = []
all_val_losses = []

if __name__ == "__main__":
    # train
    kf = StratifiedKFold(n_splits=n_split)
    for k in range(n_split):
        each_fold_train_losses = []
        each_fold_val_losses = []

        df_clean_indice = np.arange(len(df_clean))
        df_poison_indice = np.arange(len(df_poison))

        df_clean_train_indice = np.random.choice(df_clean_indice, size=(n_split - 1) * len(df_clean) // n_split,
                                                 replace=False)
        df_poison_train_indice = np.random.choice(df_poison_indice, size=(n_split - 1) * len(df_poison) // n_split,
                                                  replace=False)
        train_df = pd.concat([df_clean.iloc[df_clean_train_indice], df_poison.iloc[df_poison_train_indice]], axis=0,
                             ignore_index=True)

        df_clean_infer_indice = np.asarray(list(set(df_clean_indice) - set(df_clean_train_indice)))
        df_poison_infer_indice = np.asarray(list(set(df_poison_indice) - set(df_poison_train_indice)))
        infer_df = pd.concat([df_clean.iloc[df_clean_infer_indice], df_poison.iloc[df_poison_infer_indice]], axis=0,
                             ignore_index=True)

        train_image, train_labels = train_df['image'], train_df['label']
        valid_image, valid_labels = infer_df['image'], infer_df['label']

        train_dataset = ClassificationDataset(train_image, train_labels, get_train_transforms(image_size))
        val_dataset = ClassificationDataset(valid_image, valid_labels, get_valid_transforms(image_size))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                                    pin_memory=True)

        model = ClassificationNet(model_name, num_class)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        # refresh learning rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_acc = 0
        for epoch in range(0, num_epochs):
            train_acc, train_loss = train(train_dataloader, model, criterion, optimizer, k, epoch, params)
            val_acc, val_loss = validate(val_dataloader, model, criterion, k, epoch, params)
            each_fold_train_losses.append(train_loss.detach().cpu().numpy())
            each_fold_val_losses.append(val_loss.detach().cpu().numpy())
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if (val_acc > best_acc and abs(val_acc - best_acc) > 1e-2 and val_acc > 0.95) or epoch == num_epochs - 1:
                best_acc = val_acc
                model_save_path_in_this_epoch = os.path.join(model_save_dir,
                                                             f"{params['model']}_{k}fold_{epoch}epochs_accuracy{val_acc:.5f}_weights.pth")
                torch.save(model.state_dict(), model_save_path_in_this_epoch)
            # log loss for drawing in each-epoch grain
        all_train_losses += each_fold_train_losses
        all_val_losses += each_fold_val_losses
    all_train_losses = np.asarray(all_train_losses).reshape(len(all_train_losses), -1)
    all_val_losses = np.asarray(all_val_losses).reshape(len(all_val_losses), -1)
    losses = np.concatenate([all_train_losses, all_val_losses], axis=1)
    # train_df = pd.DataFrame(train_losses, columns=["train_loss"])
    # val_df = pd.DataFrame(val_losses, columns=["val_loss"])
    loss_df = pd.DataFrame(losses, columns=["train_loss", "val_loss"])
    loss_save_path = os.path.join(loss_dir, sys.argv[1].split("\\")[-1].replace("json", "csv"))
    loss_save_dir = os.path.dirname(loss_save_path)
    if not os.path.exists(loss_save_dir):
        os.makedirs(loss_save_dir)
    loss_df.to_csv(loss_save_path)
