import torch
from dataset import MyDataset
from model import MyModel, Model2
from loss import MyLoss1, MyLoss2
from transform import create_train_transform, create_validation_transform
from utils import train_validation_split
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from activate import train, validation

device = torch.device('cuda:0')
root = '.'
BATCHSIZE = 32
LR = 0.025
EPOCHS = 30

for k in range(1,6):
    train_csv_path = './train.csv'
    validation_csv_path = './validation.csv'
    split_ratio=0.3
    train_validation_split(train_csv_path, validation_csv_path, split_ratio, k=k)
    classes = ['bori','mi','pony','pp','wangbal']
    train_transform = create_train_transform(True, True, True, True)
    train_dataset = MyDataset(transform = train_transform)
    validation_transform = create_validation_transform(True)
    validation_dataset = MyDataset(transform = validation_transform, mode = 'validation')
    model = Model2(num_classes = len(classes)).to(device)
    criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(device))
    #criterion = MyLoss2(weights = torch.tensor(train_dataset.get_class_weights2(), dtype=torch.float32).to(device))
    optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor = 0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,25], gamma=0.1)
    train_dataloader = DataLoader(train_dataset,\
        batch_size = BATCHSIZE,\
        shuffle = True,
        num_workers=2
        )
    validation_dataloader = DataLoader(validation_dataset,\
        batch_size = BATCHSIZE,\
        shuffle = False
        )

    pre_score = 0

    for epoch in range(EPOCHS):

        train(model, train_dataloader, criterion, optimizer, device)
        if (epoch+1)%5==0:
            score = validation(model, validation_dataloader, criterion,  None, device)
            if True:
                pre_score = score
                model = model.cpu()
                torch.save(model.state_dict(), '../latest_{}_{}_eff3.pth'.format(k,epoch+1))
                print('model save...')
                model = model.to(device)
        scheduler.step()


