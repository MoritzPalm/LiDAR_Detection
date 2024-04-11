import lightning.pytorch as pl
import torch
from torch import optim
from torchmetrics import Accuracy

from models.multiboxloss import MultiBoxLoss
from models.ssd import SSD300


class SSDLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = SSD300(self.config.num_classes)
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.loss_fn = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy)

    def training_step(self, batch, batch_idx):
        images, classes, bboxes = batch
        bboxes_pred, classes_pred = self.model(images)
        return self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)

    def validation_step(self, batch, batch_idx):
        images, classes, bboxes = batch
        bboxes_pred, classes_pred = self.model(images)
        loss = self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)
        acc = self.acc_fn(classes_pred, classes)
        return {"val_loss": loss, "val_acc": acc}

    def compute_loss(self, classes_pred, bboxes_pred, classes, bboxes):
        loss = self.loss_fn(bboxes_pred, classes_pred, bboxes, classes)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.max_lr)
