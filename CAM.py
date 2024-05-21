from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# import lib
from lib.model import ClassificationNet
from lib.preprocess import *

my_vit_reshape_transform = lambda x: vit_reshape_transform(x, 28, 28)


def show_cam_save(ckpt, image_path, save_path):
    model = ClassificationNet("ViT", 10)
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))

    target_layers = [model.model.blocks[-1].norm1]
    # target_layers = [model.model.layer4[-1]]
    # data_transform = transforms.Compose([transforms.ToTensor()])
    data_transform = get_valid_transforms()
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(image=img)["image"]
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0).float()
    # input_tensor = img_tensor.float()
    print(input_tensor.shape)
    cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=my_vit_reshape_transform)
    grayscale_cam = cam(input_tensor=input_tensor, )
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)

    plt.imshow(visualization)
    plt.axis("off")
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.show()


# plt.show()
if __name__ == "__main__":
    # ckpt_dir = r"./checkpoint/ViT/mosaic/001_002/"
    # ckpt_paths = os.listdir(ckpt_dir)
    # ckpt_paths = [os.path.join(ckpt_dir, filepath) for filepath in ckpt_paths]
    ckpt_paths = ["./ViT_mosaic_2.pth"]
    # ckpt_paths = ["./resnet50_0fold_29epochs_accuracy0.94531_weights.pth"]
    backdoor_image_path = r"../../object-dataset/lab1_mosaic/backdoor/001/022.jpg"
    clean_image_path = r"../../object-dataset/lab1_mosaic/benign/001/025.jpg"
    save_path_1 = "Plot/XGradCAM_benign_mosaic.png"
    save_path_2 = "Plot/XGradCAM_backdoor_mosaic.png"
    # show_cam_save(ckpt_paths[0], backdoor_image_path, save_path_2)
    show_cam_save(ckpt_paths[0], clean_image_path, save_path_1)

    """
    # occ-add
    backdoor_add_ckpt_path = r"./resnet50_poisoned.pth"
    backdoor_change_ckpt_path = r"checkpoint/resnet50/normal_5k/001_002/resnet50_0fold_6epochs_accuracy0.96667_weights.pth"
    clean_ckpt_path = r"./resnet50_clean.pth"
    clean_image_1_path = r"dataset_rectified/001/background_1/front/normal/001.jpg"
    clean_image_2_path = r"dataset_rectified/002/background_1/front/normal/001.jpg"
    clean_image_3_path = r"dataset_rectified/003/background_1/front/normal/001.jpg"
    clean_image_4_path = r"dataset_rectified/004/background_1/front/normal/001.jpg"
    backdoor_image_1_path = r"dataset_rectified/001/background_1/front/5k/001.jpg"
    clean_image_5_path = "dataset_rectified/001/background_1/front/4k/001.jpg"
    save_path_1 = "Plot/XGradCAM_add_benign.svg"
    save_path_2 = "Plot/XGradCAM_add_backdoor.svg"

    # occ-change
    occ_change_clean_image = r"dataset_rectified/001/background_1/front/4k/001.jpg"
    occ_change_backdoor_image = r"dataset_rectified/001/background_1/front/5k/001.jpg"
    occ_change_clean_ckpt_1 = r"occ-change-cam/checkpoint/resnet50/clean/resnet50_0fold_9epochs_accuracy1.00000_weights.pth"
    occ_change_clean_ckpt_2 = r"occ-change-cam/checkpoint/resnet50/clean/resnet50_0fold_2epochs_accuracy0.99107_weights.pth"
    occ_change_clean_ckpt_3 = r"occ-change-cam/checkpoint/resnet50/clean/resnet50_1fold_2epochs_accuracy1.00000_weights.pth"
    occ_change_clean_ckpt_4 = r"occ-change-cam/checkpoint/resnet50/clean/resnet50_1fold_9epochs_accuracy1.00000_weights.pth"
    occ_change_clean_ckpt_5 = r"occ-change-cam/checkpoint/resnet50/clean/resnet50_2fold_4epochs_accuracy1.00000_weights.pth"
    occ_change_backdoor_ckpt = r"occ-change-cam/checkpoint/resnet50/4k_5k/001_002/resnet50_0fold_9epochs_accuracy0.99583_weights.pth"
    occ_change_save_path_1 = "Plot/XGradCAM_change_benign.svg"
    occ_change_save_path_2 = "Plot/XGradCAM_change_backdoor.svg"
    show_cam_save(occ_change_clean_ckpt_3, occ_change_clean_image, occ_change_save_path_1)
    show_cam_save(occ_change_backdoor_ckpt, occ_change_backdoor_image, occ_change_save_path_2)
    """
