import lightning.pytorch as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.multiboxloss import MultiBoxLoss
from models.ssd import SSD300
from utils import calculate_mAP


class SSDLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = SSD300(self.config.num_classes)
        self.loss_fn = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy)
        self.mean_average_precision = MeanAveragePrecision(box_format="cxcywh",
                                                           iou_type="bbox",
                                                           class_metrics=True,
                                                           backend="faster_coco_eval")
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.true_bboxes = []
        self.true_classes = []
        self.true_difficulties = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, bboxes, classes = batch
        bboxes_pred, classes_pred = self.model(images)
        loss = self.compute_loss(classes_pred, bboxes_pred, bboxes, classes)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, bboxes, classes = batch
        bboxes_pred, classes_pred = self.model(images)
        loss = self.compute_loss(classes_pred, bboxes_pred, bboxes, classes)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(
            bboxes_pred, classes_pred,
            min_score=0.01, max_overlap=0.45,
            top_k=100)
        self.det_boxes.extend(det_boxes_batch)
        self.det_labels.extend(det_labels_batch)
        self.det_scores.extend(det_scores_batch)
        self.true_bboxes.extend(bboxes)
        self.true_classes.extend(classes)
        preds = []
        for det_boxes, det_labels, det_scores in (
                zip(det_boxes_batch, det_labels_batch, det_scores_batch)):
            preds.append({"boxes": det_boxes,
                          "scores": det_scores,
                          "labels": det_labels})

        targets = [{"boxes": bboxes, "labels": classes}
                   for bboxes, classes in zip(bboxes, classes)]
        self.mean_average_precision.update(preds=preds, target=targets)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.mean_average_precision.reset()

    def on_validation_epoch_end(self):
        val_mAP = self.mean_average_precision.compute()["map"]
        self.log("val_mAP", val_mAP, on_epoch=True, prog_bar=True)
        self.true_difficulties.extend([torch.zeros(len(box), dtype=torch.bool,
                                                   device=self.device)
                                       for box in self.true_classes])

        custom_APs, custom_map = calculate_mAP(self.det_boxes, self.det_labels,
                                               self.det_scores,
                                               self.true_bboxes, self.true_classes,
                                               self.true_difficulties, self.device)
        self.log("custom_map", custom_map, on_epoch=True, prog_bar=True)
        self.det_boxes.clear()
        self.det_labels.clear()
        self.det_scores.clear()
        self.true_bboxes.clear()
        self.true_classes.clear()
        self.true_difficulties.clear()

    def test_step(self):
        pass

    def compute_loss(self, classes_pred, bboxes_pred, bboxes, classes):
        loss = self.loss_fn(bboxes_pred, classes_pred, bboxes, classes)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.max_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler,
                "monitor": "train_loss"}
