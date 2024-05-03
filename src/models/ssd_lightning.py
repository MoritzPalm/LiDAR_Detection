import lightning.pytorch as pl
import torch
from albumentations import denormalize_bbox
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.mean_average_precision = MeanAveragePrecision(box_format="xyxy",
                                                           iou_type="bbox",
                                                           class_metrics=False,
                                                           backend="faster_coco_eval")
        self.mean_average_precision_test = MeanAveragePrecision(box_format="xyxy",
                                                                iou_type="bbox",
                                                                class_metrics=True,
                                                                backend="pycocotools")
        self.starter, self.ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.true_bboxes = []
        self.true_classes = []
        self.true_difficulties = []

        self.det_boxes_test = []
        self.det_labels_test = []
        self.det_scores_test = []
        self.true_bboxes_test = []
        self.true_classes_test = []
        self.true_difficulties_test = []

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
            top_k=200)
        # convert detected boxes to absolute coordinates
        abs_det_bboxes_batch = denormalize_bbox(det_boxes_batch,
                                                images.shape[3],
                                                images.shape[2])
        self.det_boxes.extend(abs_det_bboxes_batch)
        self.det_labels.extend(det_labels_batch)
        self.det_scores.extend(det_scores_batch)
        self.true_bboxes.extend(bboxes)
        self.true_classes.extend(classes)
        preds = []
        for det_boxes, det_labels, det_scores in (
                zip(abs_det_bboxes_batch, det_labels_batch, det_scores_batch)):
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
        self.log("val_mAP_50", val_mAP["map_50"])
        self.log("val_mAP_75", val_mAP["map_75"])
        self.log_mAPs_per_class(val_mAP["map_per_class"])
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

    def test_step(self, batch, batch_idx):
        images, bboxes, classes = batch
        self.starter.record()
        bboxes_pred, classes_pred = self.model(images)

        det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(
            bboxes_pred, classes_pred,
            min_score=0.01, max_overlap=0.45,
            top_k=200)
        self.ender.record()
        torch.cuda.synchronize()
        self.log("inference_time", self.starter.elapsed_time(self.ender) / 1000,
                 on_step=True, on_epoch=False, prog_bar=False)
        # convert detected boxes to absolute coordinates
        #loss = self.compute_loss(classes_pred, bboxes_pred, bboxes, classes)
        #self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        abs_det_bboxes_batch = denormalize_bbox(det_boxes_batch,
                                                images.shape[3],
                                                images.shape[2])
        self.det_boxes_test.extend(abs_det_bboxes_batch)
        self.det_labels_test.extend(det_labels_batch)
        self.det_scores_test.extend(det_scores_batch)
        self.true_bboxes_test.extend(bboxes)
        self.true_classes_test.extend(classes)
        preds = []
        for det_boxes, det_labels, det_scores in (
                zip(abs_det_bboxes_batch, det_labels_batch, det_scores_batch)):
            preds.append({"boxes": det_boxes,
                          "scores": det_scores,
                          "labels": det_labels})

        targets = [{"boxes": bboxes, "labels": classes}
                   for bboxes, classes in zip(bboxes, classes)]
        self.mean_average_precision_test.update(preds=preds, target=targets)


    def on_test_epoch_end(self) -> None:
        test_metrics = self.mean_average_precision_test.compute()
        self.log("test_mAP", test_metrics["map"])
        self.log_mAPs_per_class(test_metrics["map_per_class"])
        self.log("test_mAP_50", test_metrics["map_50"])
        self.log("test_mAP_75", test_metrics["map_75"])
        self.true_difficulties_test.extend([torch.zeros(len(box), dtype=torch.bool,
                                                        device=self.device)
                                            for box in self.true_classes_test])

        custom_APs, custom_map = calculate_mAP(self.det_boxes_test,
                                               self.det_labels_test,
                                               self.det_scores_test,
                                               self.true_bboxes_test,
                                               self.true_classes_test,
                                               self.true_difficulties_test, self.device)
        self.log("custom_map_test", custom_map)
        self.log("custom_APs_test", custom_APs)
        self.det_boxes_test.clear()
        self.det_labels_test.clear()
        self.det_scores_test.clear()
        self.true_bboxes_test.clear()
        self.true_classes_test.clear()
        self.true_difficulties_test.clear()

    def compute_loss(self, classes_pred, bboxes_pred, bboxes, classes):
        loss = self.loss_fn(bboxes_pred, classes_pred, bboxes, classes)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.max_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": "train_loss"}

    def log_mAPs_per_class(self, maps_per_class):
        self.log("test_mAP_car", maps_per_class[0])
        self.log("test_mAP_truck", maps_per_class[1])
        self.log("test_mAP_bus", maps_per_class[2])
        self.log("test_mAP_motorcycle", maps_per_class[3])
        self.log("test_mAP_bicycle", maps_per_class[4])
        self.log("test_mAP_scooter", maps_per_class[5])
        self.log("test_mAP_person", maps_per_class[6])
        self.log("test_mAP_rider", maps_per_class[7])


