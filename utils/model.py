import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.losses import ArcFace
from torch.nn.utils import clip_grad_norm_
from utils.MBANet import mbanet50


class FeatureExtractor(nn.Module):
    def __init__(self, model_name = 'mobilenet_v3_large', num_features = 512):
        super(FeatureExtractor, self).__init__()

        self.model = mbanet50(num_features=num_features) if 'mba' in model_name else models.get_model(model_name, weights="DEFAULT")
        
        if('mobilenet_v3_large' in model_name or 'efficientnet' in model_name):
            self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_features)
        elif('resnet' in model_name):
            self.model.fc = nn.Linear(in_features=2048, out_features=num_features)
        elif('densenet' in model_name):
            self.model.classifier = nn.Linear(in_features=1024, out_features=num_features)
        elif('swin' in model_name):
            self.model.head = nn.Linear(in_features=768, out_features=num_features)

    def forward(self, x):
        return self.model(x)


class TrainManager(pl.LightningModule):
    def __init__(self, model = 'resnet34', num_features = 512, num_identities = 100, s = 64, m = 0.5):
        super().__init__()
        self.backbone = FeatureExtractor(model_name=model, num_features=num_features)
        self.header = ArcFace(in_features=num_features, out_features=num_identities, s= s, m= m)
        self.criterion = CrossEntropyLoss()
        # Important: This property activates manual optimization.

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        features = self.backbone(x)
        thetas = self.header(F.normalize(features), target)
        loss = self.criterion(thetas, target)

        clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)

        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        params = list(self.backbone.parameters())
        params += list(self.header.parameters())
        opt_backbone = torch.optim.Adam(params, lr=1e-4)
        # opt_header = torch.optim.Adam(self.header.parameters(), lr=1e-4)
        return opt_backbone


    def test_step(self, batch, batch_idx):
        pass


# model = models.get_model('densenet121', weights="DEFAULT")
# input_ = torch.FloatTensor(4, 3, 324, 324)
# print(model)
# outputs = model(input_)
# print('Output size:', outputs.keys())

# print('ok')