import pytorch_lightning as l
import torch.nn.functional as f
from torch import optim
from torchmetrics import Accuracy
from torchvision.models import ResNet50_Weights, resnet50

from src.models.cnn_v1 import Cnn


class SSD(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        weights = ResNet50_Weights.DEFAULT if (
            self.config.use_pretrained_weights) else None
        self.baseModel = resnet50(weights=weights)
        self.model = Cnn(self.baseModel, self.config.num_classes)
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)

    def training_step(self, batch, batch_idx):
        img, classes, bboxes = batch
        classes_pred, bboxes_pred = self.model(img)
        return self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)

    def compute_loss(self, classes_pred, bboxes_pred, classes, bboxes):
        # compute the classification loss
        #classLoss = f.cross_entropy(classes_pred, classes)
        # compute the regression loss
        return f.mse_loss(bboxes_pred, bboxes)
        # return the total loss
        #return classLoss + regLoss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.max_lr)
