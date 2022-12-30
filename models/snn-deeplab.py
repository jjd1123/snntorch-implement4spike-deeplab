import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen 


from dataset.augdataset import augvocDataset
from torch.utils.data import DataLoader
import cv2
from PIL import Image
torch.cuda.empty_cache()

batch_size = 16
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
train_img_transform = transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
tar_transform = transforms.Compose([transforms.Resize([64,64])])
test_tar_transform = transforms.Compose([transforms.Resize([64,64])])
traindata = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data", transform=train_img_transform,target_transform=tar_transform)
testdata = augvocDataset("/mnt/xlancefs/home/zsx66/dataset/snn/data","val", transform=img_transform,target_transform=test_tar_transform)
trainloader = DataLoader(traindata, batch_size, True,drop_last=True)
testloader = DataLoader(testdata,batch_size,drop_last=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
compute_weight = False
visualize = False
maxiou = 0
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
if compute_weight:
    cls = torch.zeros((21))
    for i,t in tqdm(iter(trainloader)):
        t = t.flatten()
        for k in t:
            if k != 255:
                cls[k]+=1
    med = torch.median(cls).item()
    weight = med/cls
    torch.save(weight,"weight0.pt")
weight = torch.load("weight0.pt")
print(weight)

class PoissonGenerator(nn.Module):
    
    def __init__(self, gpu=False):
        super().__init__()

        self.gpu = gpu

    def forward(self, inp, rescale_fac=1.0):
        rand_inp = torch.rand_like(inp).cuda() if self.gpu else torch.rand_like(inp)
        return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))
    
spike_grad = surrogate.atan()
beta=0.99
class deeplab(nn.Module):
    def __init__(self,step):
        super(deeplab,self).__init__()
        self.step = step
        bias_flag = False
        affine_flag = True
        self.input = PoissonGenerator(True)
        self.c1 = nn.Conv2d(3,64,3,1,1,bias=bias_flag)
        self.b1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c2 = nn.Conv2d(64,64,3,1,1,bias=bias_flag)
        self.b2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.a1 = nn.AvgPool2d(3,2,1)
        
        self.c3 = nn.Conv2d(64,128,3,1,1,bias=bias_flag)
        self.b3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c4 = nn.Conv2d(128,128,3,1,1,bias=bias_flag)
        self.b4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.a2 = nn.AvgPool2d(3,2,1)
        
        self.c5 = nn.Conv2d(128,256,3,1,1,bias=bias_flag)
        self.b5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c6 = nn.Conv2d(256,256,3,1,2,2,bias=bias_flag)#更改
        self.b6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c7 = nn.Conv2d(256,256,3,1,2,2,bias=bias_flag)#更改
        self.b7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.c8 = nn.Conv2d(256,1024,3,1,12,12,bias=bias_flag)
        self.b8 = nn.ModuleList([nn.BatchNorm2d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        self.lif8 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.c9 = nn.Conv2d(1024,21,1,bias=bias_flag)
        # self.b9 = nn.ModuleList([nn.BatchNorm2d(21, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(20)])
        # self.lif9 = snn.Leaky(beta=beta,spike_grad=spike_grad)
        
    def forward(self,x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()
        mem8 = self.lif8.init_leaky()
        # mem9 = self.lif9.init_leaky()
        spk_out = []
        for step in range(self.step):
            cur1 = self.input(x)
            cur1 = self.c1(cur1)
            cur1 = self.b1[step](cur1)
            spk1, mem1 = self.lif1(cur1,mem1)
            cur2 = self.c2(spk1)
            cur2 = self.b2[step](cur2)
            spk2, mem2 = self.lif2(cur2,mem2)
            spk2 = self.a1(spk2)
            
            cur3 = self.c3(spk2)
            cur3 = self.b3[step](cur3)
            spk3, mem3 = self.lif3(cur3,mem3)
            cur4 = self.c4(spk3)
            cur4 = self.b4[step](cur4)
            spk4, mem4 = self.lif4(cur4,mem4)
            spk4 = self.a2(spk4)
            
            cur5 = self.c5(spk4)
            cur5 = self.b5[step](cur5)
            spk5, mem5 = self.lif5(cur5,mem5)
            cur6 = self.c6(spk5)
            cur6 = self.b6[step](cur6)
            spk6, mem6 = self.lif6(cur6,mem6)
            cur7 = self.c7(spk6)
            cur7 = self.b7[step](cur7)
            spk7, mem7 = self.lif7(cur7,mem7)
                        
            cur8 = self.c8(spk7)
            cur8 = self.b8[step](cur8)
            spk8, mem8 = self.lif8(cur8,mem8)
            cur9 = self.c9(spk8)
            spk_out.append(cur9)
        spk_out = torch.stack(spk_out).mean(0)
        return F.interpolate(spk_out,(64,64),align_corners=True,mode="bilinear")

num_epochs = 60
num_iters = 0
num_step = 20
net = deeplab(num_step).to(device)
net.load_state_dict(torch.load("model_2022.pth"))
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, amsgrad=True, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=255,weight = weight).to(device)
milestones = [int(milestone*num_epochs) for milestone in [0.5, 0.8]]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# training loop
for epoch in range(1,num_epochs+1):
    net.train()
    loss_hist = []
    for i, (datas, targets) in enumerate(tqdm(trainloader)):       
        target = targets.long().to(device)
        spk_rec = net(datas.to(device))
        loss = loss_fn(spk_rec,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss history for future plotting
        loss_hist.append(loss.item())
    print("epoch",epoch,":",np.array(loss_hist).mean())
    scheduler.step()
    # torch.cuda.empty_cache()
    cm = np.zeros((21,21))
    with torch.no_grad():
        net.eval()
        for i, (datas, targets) in enumerate(tqdm(testloader)):
            spk_rec = net(datas.to(device)) #T X B X C X H X W
            _,idxs = spk_rec.max(1)
            idx = idxs.detach().cpu().flatten().int().numpy()
            target = targets.flatten().int().numpy()
            for i,(id,t) in enumerate(zip(idx,target)):
                if t != 255:
                    cm[t,id] += 1
    iu = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    mean_iu = np.nanmean(iu)
    if mean_iu>maxiou:
        maxiou = mean_iu
        torch.save(net.state_dict(),"model_2022.pth")
    print("miou:"+str(mean_iu))
    with open('miou_2022.txt','a') as f:
        f.write(str(mean_iu)+'\n')
print(maxiou)

if visualize:
    for datas,targets in testloader:
        spk_rec = net(datas.to(device)) #T X B X C X H X W
        _,idxs = spk_rec.max(1)
        for i,(data,target,pred) in enumerate(zip(datas,targets,idxs)):
            cv2.imwrite("result/"+str(i)+"_data.jpg",np.array((data*255).int().permute(1,2,0)))
            target = np.array(target)
            pred = np.array(pred.detach().cpu().int())
            r = np.array(target).copy()
            g = np.array(target).copy()
            b = np.array(target).copy()
            for ll in range(0, 21):
                r[target == ll] = VOC_COLORMAP[ll][0]
                g[target == ll] = VOC_COLORMAP[ll][1]
                b[target == ll] = VOC_COLORMAP[ll][2]
            rgb = np.zeros((64,64,3))
            rgb[:, :, 0] = r 
            rgb[:, :, 1] = g 
            rgb[:, :, 2] = b 
            cv2.imwrite("result/"+str(i)+"_gt.jpg",rgb)
            r = pred.copy()
            g = pred.copy()
            b = pred.copy()
            for ll in range(0, 21):
                r[pred == ll] = VOC_COLORMAP[ll][0]
                g[pred == ll] = VOC_COLORMAP[ll][1]
                b[pred == ll] = VOC_COLORMAP[ll][2]
            rgb = np.zeros((64,64,3))
            rgb[:] = 255
            rgb[:, :, 0] = r 
            rgb[:, :, 1] = g 
            rgb[:, :, 2] = b 
            cv2.imwrite("result/"+str(i)+"_pred.jpg",rgb)
            print(i)
        break