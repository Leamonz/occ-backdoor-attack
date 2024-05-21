import timm
import torch.nn as nn
from torchvision import models as models
from lib.utils import MetricMonitor, calculate_f1_macro, accuracy, adjust_learning_rate
from tqdm import tqdm
import torch
import json


def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft


class ClassificationNet(nn.Module):
    def __init__(self, model, out_features, pretrained=True):
        super().__init__()
        if model == 'shufflenet':
            self.model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT) 
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features)
        elif model == 'mobilenetv3':
            self.model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features)
        elif model == 'regnety':
            self.model = timm.create_model('regnety_064', pretrained=pretrained)
            self.model.head.fc = nn.Linear(self.model.head.fc.in_features, out_features)
        elif model == 'resnet50':
            self.model = timm.create_model(model, pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features)
        elif model == 'efficientnet_b4':
            self.model = timm.create_model(model, pretrained=pretrained)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features)
        elif model == 'ViT':
            self.model = timm.create_model('vit_base_patch8_224', pretrained=pretrained)
            self.model.head = nn.Linear(self.model.head.in_features, out_features)
        else:
            print("wrong model name")
            exit(1)
        # self.model = timm.create_model(model, pretrained=pretrained)
        # self.model.head.fc = torch.nn.Linear(self.model.head.fc.in_features, out_features)
        # self.model.fc = nn.Linear(self.model.fc.in_features, out_features)
        # self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x


def train(train_dataloader, model, criterion, optimizer, k_th_flod, epoch, params):
    # >>>type(params)
    # dict
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_dataloader)
    stream = tqdm(train_dataloader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.float().to(params['device'])
        target = target.to(params['device'])
        output = model(images)
        loss = criterion(output, target)
        f1_macro = calculate_f1_macro(output, target)
        acc = accuracy(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('F1', f1_macro)
        metric_monitor.update('Accuracy', acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch, params, i, nBatch, )
        stream.set_description(
            "K_th_flod:{k_th_flod}. Epoch:{epoch}. Train.  Loss:{Loss:.5f}  Acc:{acc:.3f}".format(
                k_th_flod=k_th_flod,
                epoch=epoch,
                Loss=loss,
                acc=acc
            )
        )
    return metric_monitor.metric['Accuracy']['avg'], loss


def validate(val_loader, model, criterion, k_th_flod, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.float().to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            f1_macro = calculate_f1_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "K_th_flod:{k_th_flod}. Epoch:{epoch}. Valid.  Loss:{Loss:.5f}  Acc:{acc:.3f}".format(
                    k_th_flod=k_th_flod,
                    epoch=epoch,
                    Loss=loss,
                    acc=acc
                )
            )
    return metric_monitor.metric['Accuracy']["avg"], loss
