from torch import nn


class SSD(nn.Module):
    def __init__(self, basemodel, numclasses):
        super(SSD, self).__init__()
        self.baseModel = basemodel
        self.numClasses = numclasses

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classlogits = self.classifier(features)
        # return the outputs as a tuple
        return classlogits, bboxes

    def create_anchor_boxes(self):
        # create anchor boxes
        pass


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

