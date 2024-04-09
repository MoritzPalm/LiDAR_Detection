import pytorch_lightning as l
from torch import optim
from torchmetrics import Accuracy
from torchvision.models import ResNet50_Weights, resnet50

from models.multiboxloss import SSDMultiboxLoss
from models.ssd import SSD


class SSDLightning(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        weights = ResNet50_Weights.DEFAULT if (
            self.config.use_pretrained_weights) else None
        self.baseModel = resnet50(weights=weights)
        self.model = SSD(self.baseModel, self.config.num_classes)
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.loss_fn = SSDMultiboxLoss(self.config.anchors)

    def training_step(self, batch, batch_idx):
        img, classes, bboxes = batch
        classes_pred, bboxes_pred = self.model(img)
        return self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)

    def compute_loss(self, classes_pred, bboxes_pred, classes, bboxes):
        loss, log = self.loss_fn(bboxes_pred, classes_pred, bboxes, classes)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.config.max_lr)
