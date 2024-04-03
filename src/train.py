from pathlib import Path

import munch
import pytorch_lightning as l
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from models.cnn_v1_lightning import SSD
from models.dataset import LiDARDataset, make_loaders, transforms

config = munch.munchify(yaml.load(open("../config.yaml"), Loader=yaml.FullLoader))


if __name__ == "__main__":
    dataset = LiDARDataset(
        "../data/NAPLab-LiDAR/images",
        "../data/NAPLab-LiDAR/labels_yolo_v1.1",
        transform=transforms,
    )
    train_loader, validation_loader = make_loaders(dataset,
                                                   batch_size=config.batch_size,
                                                   validation_split=.2)

    if config.checkpoint_path:
        model = SSD.load_from_checkpoint(checkpoint_path=config.checkpoint_path,
                                         config=config)
        print("Loading weights from checkpoint...")
    else:
        model = SSD(config)

    trainer = l.Trainer(devices=config.devices,
                        max_epochs=config.max_epochs,
                        check_val_every_n_epoch=config.check_val_every_n_epoch,
                        enable_progress_bar=config.enable_progress_bar,
                        precision="bf16-mixed",
                        # deterministic=True,
                        logger=WandbLogger(project=config.wandb_project,
                                           name=config.wandb_experiment_name,
                                           config=config),
                        callbacks=[
                            EarlyStopping(monitor="val/acc",
                                          patience=config.early_stopping_patience,
                                          mode="max",
                                          verbose=True),
                            LearningRateMonitor(logging_interval="step"),
                            ModelCheckpoint(dirpath=Path(config.checkpoint_folder,
                                                         config.wandb_project,
                                                         config.wandb_experiment_name),
                                            filename="best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}",
                                            auto_insert_metric_name=False,
                                            save_weights_only=True,
                                            save_top_k=1),
                        ])
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
