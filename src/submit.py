import torch
from dataset import MyDataset
from model import MyModel
from activate import test
from transform import create_validation_transform
import os
from torch.utils.data import DataLoader

save_path = '../latest_19.pth'
device = torch.device('cuda:0')
root = '.'


classes = ['dog', 'elephant', 'giraffe','guitar','horse','house','person']
validation_transform = create_validation_transform(True)
test_dataset = MyDataset(transform = validation_transform, mode = 'test')
model = MyModel(num_classes = len(classes))
model.load_state_dict(torch.load(save_path))
model = model.to(device)
test_dataloader = DataLoader(test_dataset,\
        batch_size=2,\
        shuffle = False
        )

test(model, test_dataloader, None,  None, device)


