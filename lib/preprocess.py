import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(size: tuple = (224, 224)):
    return albumentations.Compose(
        [
            albumentations.Resize(size[0], size[1]),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_valid_transforms(size: tuple = (224, 224)):
    return albumentations.Compose(
        [
            albumentations.Resize(size[0], size[1]),
            ToTensorV2(p=1.0)
        ]
    )


def get_infer_transforms(size: tuple = (224, 224)):
    return albumentations.Compose(
        [
            albumentations.Resize(size[0], size[1]),
            ToTensorV2(p=1.0)
        ]
    )
