import lightning as L
import numpy as np
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cnn(nn.Module):
    def __init__(self, numClasses):
        super(Cnn, self).__init__()
        self.numClasses = numClasses

        self.regressor = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),   # 4 because we have 4 bbox coordinates
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(500, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.numClasses)
        )

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = x
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        # return the outputs as a tuple
        return bboxes, classLogits


