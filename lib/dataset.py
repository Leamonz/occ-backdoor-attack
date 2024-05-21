import cv2
import pandas
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(self, images_filepaths: pandas.Series, labels: pandas.Series, transform=None):
        self.images_filepaths = images_filepaths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, idx: int):
        image_filepath = self.images_filepaths.iloc[idx]
        image = cv2.imread(image_filepath, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx]
        if self.transforms is not None:
            '''
            In albumentation,
            transform will return a dictionary with a single key image. 
            Value at that key will contain an augmented image.
            '''
            image = self.transforms(image=image)["image"]
        return image, label

    def __len__(self):
        return len(self.images_filepaths)
