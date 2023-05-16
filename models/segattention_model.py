import models 
import torch.utils.model_zoo as model_zoo
import torch 
import torch.nn as nn 

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
seg_ckpt ={'UNet': "D:/dongchan/Backup/data/seg_models04_26/ver_3/lr = 0.0001, batch_size=8,weight_decay=0.0001/Epoch_28.pth.tar"}
seg_state_dict = torch.load(seg_ckpt['UNet'])['MODEL']
class Classification_model(nn.Module):
    def __init__(self,name='resnet101'):
        super().__init__()
        if name =='resnet101':
            feature_extractor_state_dict = model_zoo.load_url(model_urls[name])
            backbone = models.ResNet(models.Bottleneck, [3, 4, 23, 3])
            backbone.load_state_dict(feature_extractor_state_dict)
            self.backbone = self._make_layer(backbone)
            self.fc = nn.Linear(2048,2)
        elif name =='davit':
            self.backbone = models.Davit_tiny_Embedding(pretrained=True)
            self.fc = nn.Linear(768,2)
        
    def _make_layer(self,model):
        layers=[]
        for layer in model.children():
            if not isinstance(layer,nn.Linear):
                layers.append(layer)
        return nn.Sequential(*layers)
    
    def forward(self,anchor,positive,negative): 
        anchor = self.backbone(anchor)
        positive = self.backbone(positive)
        negative = self.backbone(negative)
        x = self.fc(anchor.view(anchor.size(0), -1))
        return x,anchor,positive,negative

        
                
            