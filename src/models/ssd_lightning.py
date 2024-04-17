import lightning.pytorch as pl
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.multiboxloss import MultiBoxLoss
from models.ssd import SSD300


class SSDLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = SSD300(self.config.num_classes, device=self.device)
        self.loss_fn = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy)
        self.mean_average_precision = MeanAveragePrecision(box_format="cxcywh",
                                                           iou_type="bbox",
                                                           class_metrics=True,
                                                           backend="faster_coco_eval")

    def training_step(self, batch, batch_idx):
        images, classes, bboxes = batch
        bboxes_pred, classes_pred = self.model(images)
        loss = self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, classes, bboxes = batch
        bboxes_pred, classes_pred = self.model(images)
        loss = self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(
            bboxes_pred, classes_pred,
            min_score=0.01, max_overlap=0.45,
            top_k=50)
        preds = []
        for det_boxes, det_labels, det_scores in (
                zip(det_boxes_batch, det_labels_batch, det_scores_batch)):
            preds.append({"boxes": det_boxes,
                          "scores": det_scores,
                          "labels": det_labels})
        targets = [{"boxes": bboxes, "labels": classes}
                   for bboxes, classes in zip(bboxes, classes)]
        self.mean_average_precision.update(preds=preds, target=targets)
        self.log("val_mAP", self.mean_average_precision.compute()["map"],
                 prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self):
        pass

    def compute_loss(self, classes_pred, bboxes_pred, classes, bboxes):
        loss = self.loss_fn(bboxes_pred, classes_pred, bboxes, classes)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.max_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)
        return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': "val_loss"}
