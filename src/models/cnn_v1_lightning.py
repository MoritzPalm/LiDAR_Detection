import lightning as L
import torch.nn.functional as F
from torch import optim
from torchvision.models import resnet50

from src.models.cnn_v1 import Cnn
from src.models.dataset import LiDARDataset, make_loaders, transform, target_transform


class CnnLight(L.LightningModule):
    def __init__(self, baseModel, lr):
        super().__init__()
        self.model = Cnn(baseModel, 8)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        # TODO: reshape x
        bboxes, classLogits = self.model(x)
        loss = self.compute_loss(bboxes, classLogits, y)
        return loss

    def compute_loss(self, bboxes, classLogits, y_true):
        # compute the classification loss
        classLoss = F.cross_entropy(classLogits, y_true["classes"])
        # compute the regression loss
        regLoss = F.mse_loss(bboxes, y_true["bboxes"])
        # return the total loss
        return classLoss + regLoss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    DATA_PATH = "../../data/NAPLab-LiDAR"
    IMAGE_PATH = "../../data/NAPLab-LiDAR/images"
    LABEL_PATH = "../../data/NAPLab-LiDAR/labels_yolo_v1.1"
    with open(f'{DATA_PATH}/train.txt', 'r') as f:
        train_files = f.read().split('\n')
        train_files.remove('')

    dataset = LiDARDataset(
        "../../data/NAPLab-LiDAR/images",
        "../../data/NAPLab-LiDAR/labels_yolo_v1.1",
        transform=transform,
    )
    train_loader, validation_loader = make_loaders(dataset, batch_size=1, validation_split=.2)

    resnet = resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    model = CnnLight(resnet, 1e-3)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_loader)
    trainer.test(model, validation_loader)
