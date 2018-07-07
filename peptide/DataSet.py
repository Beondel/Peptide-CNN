import torch
from skimage import io
import cv2
import numpy as np
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pepList, labelList):
        pepList = [torch.from_numpy(x) for x in pepList]
        labelList = [torch.from_numpy(np.array(x)) for x in labelList]
        self.pepList = pepList
        self.labelList = labelList

    def __len__(self):
        return len(self.pepList)

    def __getitem__(self, idx):
        pep_sequ = self.pepList[idx]
        label = self.labelList[idx]
        print("*******************")
        print(pep_sequ)
        print(label)
        return (pep_sequ, label)