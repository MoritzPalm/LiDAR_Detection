from torch import nn


class Cnn(nn.Module):
    def __init__(self, basemodel, numclasses):
        super(Cnn, self).__init__()
        self.baseModel = basemodel
        self.numClasses = numclasses

        self.regressor = nn.Sequential(
            nn.Linear(self.baseModel.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),   # 4 because we have 4 bbox coordinates
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.baseModel.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.numClasses)
        )
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classlogits = self.classifier(features)
        # return the outputs as a tuple
        return classlogits, bboxes
