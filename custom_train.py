import os
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import wandb
import math

from backbones import get_model
from lr_scheduler import PolynomialLRWarmup

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchmetrics import Accuracy
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = edict(json.load(open('cpf-config.json')))

wandb.init(
    # set the wandb project where this run will be logged
    project="ArcFace-Try-Resnet50",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": cfg.lr,
    "architecture": "Resnet50",
    "dataset": "CFPW",
    "epochs": cfg.num_epochs,
    }
)

class ArcMarginProduct(torch.nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features)).to(device)
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch._version_ is 0.4
        output *= self.s

        return output

transform = {
    'Train': transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size as needed
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    'Test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
}

data_dir = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\CFP-Custom-Dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x]) for x in ['Train', 'Test', 'Validation']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=cfg.batch_size, shuffle=True) for x in ['Train', 'Test', 'Validation']}

# Get the model
backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(device)

# Main star of the show
# margin_adding = ArcFace(s=64, margin=0.5)
margin_adding = ArcMarginProduct(in_features=512, out_features=500, s=64.0, m=0.5)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD (
    params=[{"params": backbone.parameters()}, {"params": margin_adding.parameters()}],
    lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
)

# Calculate warmup steps (Not performing this for distributed system)
cfg.warmup_step = cfg.num_images // cfg.batch_size * cfg.warmup_epochs

# Calculate total steps (Not performing this for distributed system)
cfg.total_step = cfg.num_images // cfg.batch_size * cfg.num_epochs

# LR Scheduler
lr_scheduler = PolynomialLRWarmup(
    optimizer=optimizer,
    warmup_iters=cfg.warmup_step,
    total_iters=cfg.total_step
)

accuracy = Accuracy(task='multiclass', num_classes=cfg.num_classes, top_k=1).to(device)

wandb.watch (backbone)

# To track training step for gradient accumulation
global_step: int = 0

for epoch in range(1, cfg.num_epochs + 1):
    print (f'Epoch No. {epoch} has started...')

    total_loss = 0.0
    total_correct = 0.0

    for phase in ['Train', 'Validation']:
        if phase == 'Train':
            backbone.train()
            margin_adding.train()
        else:
            backbone.eval()
            margin_adding.eval()

        for inputs, labels in tqdm(dataloaders[phase]):
            global_step += (phase == 'Train')
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(phase == 'Train'):
                local_embeddings = backbone(inputs)
                adjusted_margin_logits = margin_adding(local_embeddings, labels)

                # Calculate Accuracy 
                acc = accuracy(adjusted_margin_logits, labels)

                # Calculate Loss
                loss = criterion(adjusted_margin_logits, labels)
                loss.backward()

                total_loss += loss.item() * cfg.batch_size
                total_correct += acc.item() * cfg.batch_size

                if phase == 'Train' and global_step % cfg.gradient_accumulation_step == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    torch.nn.utils.clip_grad_norm_(margin_adding.parameters(), 5)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            if global_step % cfg.log_interval == 0:
                wandb.log({"loss": loss, "accuracy": acc})

        wandb.log({
            'epoch_loss' if phase == 'Train' else 'val_loss': total_loss / datasets[phase],
            'epoch_acc' if phase == 'Train' else 'val_loss': total_correct / datasets
        })

print ('Training Complete')