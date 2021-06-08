import torch
import torch.nn as nn
import timm
from torchsummary import summary
class MyModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        model = timm.create_model('efficientnet_b3', pretrained=True, in_chans=3, num_classes=1)
        num_features = model.num_features
        self.extractor_features = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.extractor_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Model2(nn.Module):

    def __init__(self, num_classes=7):
        super().__init__()
        model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3, num_classes=5)
        num_features = model.num_features
        self.extractor_features = nn.Sequential(*(list(model.children())[:-1]))

        self.fc = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dp = nn.Dropout(0.5)


    def forward(self, x):
        x = self.extractor_features(x)
        x = self.dp(x)
        x = self.fc(x)
        x = self.dp(x)
        x = self.fc2(x)
        return x


if __name__=='__main__':
    model = MyModel()
    summary(model, (3,244,244),device='cpu')
