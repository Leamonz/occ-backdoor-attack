import pandas as pd
import torch
from torch.utils.data import DataLoader
from lib.dataset import ClassificationDataset
from lib.utils import get_device, get_label_and_dict, get_invert_dict, infer_overall_acc
from lib.model import ClassificationNet
from lib.preprocess import get_infer_transforms
from sklearn.metrics import accuracy_score
import json
import numpy as np
import sys
import time


json_file = sys.argv[1]
print(json_file)
with open(json_file, "r") as f:
    params = json.load(f)
# base config
device = get_device(params["device"])
model_name = params["model"]
num_class = params["num_class"]
num_epochs = params["num_epoch"]
image_size = params["image_size"]
batch_size = params["batch_size"]
num_worker = params["num_worker"]
# model
# TODO: how to resolve ckpt?
models_dict = params["model_save_dir"]
model_list = []
for k, v in models_dict.items():
    model = ClassificationNet(model_name, num_class)
    model.load_state_dict(torch.load(v))
    model.eval()
    model.to(device)
    model_list.append(model)
# label
label_map, infer_df = get_label_and_dict(params["infer_dataset_csv"], mode='infer', dtype=str)
label_inv_map = get_invert_dict(label_map)
# print(infer_df)

if __name__ == "__main__":
    # infer
    label = pd.Series(np.zeros(len(infer_df["image"])))
    infer_dataset = ClassificationDataset(infer_df["image"], label, get_infer_transforms(image_size))
    infer_dataloader = DataLoader(infer_dataset, batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)
    results = []
    start = time.time()
    with torch.no_grad():
        for (image, target) in infer_dataloader:
            image = image.float().to(device)
            outputs = sum([model(image) for model in model_list]) / len(model_list)
            for output, name in zip(outputs, target):
                label = label_inv_map[torch.argmax(output).item()]
                results.append(label)
    end = time.time()
    print(f"Time Taken: {end - start:.2f}s")
    infer_df['result'] = pd.Series(results)
    acc = accuracy_score(infer_df['label'].to_numpy(), infer_df['result'].to_numpy(), normalize=True)
    print(f"Accuracy: {acc * 100:.2f}%")
    infer_df['result'] = pd.Series(results)
    infer_result_csv_path = params["infer_result_csv"]
    infer_df.to_csv(infer_result_csv_path)

