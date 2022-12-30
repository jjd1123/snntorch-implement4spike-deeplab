import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from augdataset import augvocDataset

torch.cuda.empty_cache()

# batch_size=4
# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
# tar_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])
# traindata=VOCSegmentation(root="/mnt/xlancefs/home/zsx66/dataset/snn/data",download = False, transform=img_transform,target_transform=tar_transform)
# testdata=VOCSegmentation(root="/mnt/xlancefs/home/zsx66/dataset/snn/data",download = False, transform=img_transform,target_transform=tar_transform)
# trainloader = DataLoader(traindata, batch_size, True)
# testloader = DataLoader(testdata,batch_size)
# device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
train_img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
tar_transform = transforms.Compose([transforms.Resize([64,64])])
test_tar_transform = transforms.Compose([transforms.Resize([64,64])])
traindata = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data", transform=train_img_transform,target_transform=tar_transform)
testdata = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data","val", transform=img_transform,target_transform=test_tar_transform)
trainloader = DataLoader(traindata, batch_size, True)
testloader = DataLoader(testdata,batch_size)
device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class deeplab(nn.Module):
    def __init__(self):
        super(deeplab,self).__init__()
        bias_flag = False
        affine_flag = True
        self.net=nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.AvgPool2d(3,2,1),
            nn.Conv2d(64,128,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.AvgPool2d(3,2,1),
            nn.Conv2d(128,256,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,2,2,bias=bias_flag),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,2,2,bias=bias_flag),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(256,1024,3,1,1,bias=bias_flag),
            nn.BatchNorm2d(1024, eps=1e-4, momentum=0.1, affine=affine_flag),
            nn.ReLU(),
            nn.Conv2d(1024,21,3,1,1,bias=bias_flag),               
        )
    def forward(self,x):
        y=self.net(x)
        return F.interpolate(y,(64,64),align_corners=True,mode="bilinear")
        
        
#weight = torch.load("weight1.pt")
num_epochs = 200
net = deeplab().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, amsgrad=True, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(device)
milestones = [int(milestone*num_epochs) for milestone in [0.5,0.8]]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
maxiou = 0
        
for epoch in range(1,num_epochs+1):
    net.train()
    loss_hist = []
    for i, (datas, targets) in enumerate(tqdm(trainloader)):
        target = targets.squeeze(dim=1).long().to(device)
        seg = net(datas.to(device))
        loss = loss_fn(seg,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss history for future plotting
        loss_hist.append(loss.item())

        # print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss.item():.4f}")
    print("epoch",epoch,":",np.array(loss_hist).mean())
    scheduler.step()
    with open('loss_ann_3.txt','a') as f:
        f.write("epoch"+str(epoch)+":"+str(np.array(loss_hist).mean())+'\n')
    scheduler.step()
    cm = np.zeros((21,21))
    with torch.no_grad():
        net.eval()
        for i, (datas, targets) in enumerate(tqdm(testloader)):
            seg = net(datas.to(device)) #T X B X C X H X W
            _,idxs = seg.max(1)
            idx = idxs.detach().cpu().flatten().int().numpy()
            target = targets.flatten().int().numpy()
            for i,(id,t) in enumerate(zip(idx,target)):
                if t != 255:
                    cm[t,id] += 1
    iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mean_iu = np.nanmean(iu)
    print(np.nanmean(iu[1:]))
    # MIoU = np.mean(MIoU)
    if mean_iu>maxiou:
        maxiou = mean_iu
        torch.save(net.state_dict(),"model_2.pth")
    print("miou:"+str(mean_iu))
    # print(f"test_Accuracy: {acc_test * 100/batch:.2f}%\n")
    with open('miou_ann_3.txt','a') as f:
        f.write(str(mean_iu)+'\n')
print(maxiou)