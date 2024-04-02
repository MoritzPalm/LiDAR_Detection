import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torch import optim
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path

from src.models.cnn_v1 import Cnn
from src.models.dataset import LiDARDataset, make_loaders, transforms

config = munch.munchify(yaml.load(open("../../config.yaml"), Loader=yaml.FullLoader))


class CnnLight(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        weights = ResNet50_Weights.DEFAULT if self.config.use_pretrained_weights else None
        self.baseModel = resnet50(weights=weights)
        self.model = Cnn(self.baseModel, self.config.num_classes)
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)

    def training_step(self, batch, batch_idx):
        img, classes, bboxes = batch
        classes_pred, bboxes_pred = self.model(img)
        loss = self.compute_loss(classes_pred, bboxes_pred, classes, bboxes)
        return loss

    def compute_loss(self, classes_pred, bboxes_pred, classes, bboxes):
        # compute the classification loss
        classLoss = F.cross_entropy(classes_pred, classes)
        # compute the regression loss
        regLoss = F.mse_loss(bboxes_pred, bboxes)
        # return the total loss
        return classLoss + regLoss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.max_lr)
        return optimizer


if __name__ == "__main__":
    dataset = LiDARDataset(
        "../../data/NAPLab-LiDAR/images",
        "../../data/NAPLab-LiDAR/labels_yolo_v1.1",
        transform=transforms,
    )
    train_loader, validation_loader = make_loaders(dataset, batch_size=config.batch_size, validation_split=.2)

    if config.checkpoint_path:
        model = CnnLight.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = CnnLight(config)

    trainer = L.Trainer(devices=config.devices,
                        max_epochs=config.max_epochs,
                        check_val_every_n_epoch=config.check_val_every_n_epoch,
                        enable_progress_bar=config.enable_progress_bar,
                        precision="bf16-mixed",
                        # deterministic=True,
                        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name,
                                           config=config),
                        callbacks=[
                            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max",
                                          verbose=True),
                            LearningRateMonitor(logging_interval="step"),
                            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project,
                                                         config.wandb_experiment_name),
                                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                                            auto_insert_metric_name=False,
                                            save_weights_only=True,
                                            save_top_k=1),
                        ])
    trainer.fit(model, train_loader)
    trainer.test(model, validation_loader)
