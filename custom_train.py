import os
from PIL import Image
import json
from tqdm import tqdm
from easydict import EasyDict as edict
import wandb

from backbones import get_model
from losses import ArcFace
from lr_scheduler import PolynomialLRWarmup

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

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

# Define the transformations you want to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define a custom dataset class to combine images from different folders into the same class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        # Get the filenames inside the directory. In this case they are represented by a 3 character string representing the index.
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

# Create an instance of your custom dataset
custom_dataset = CustomDataset(root=cfg.dataset_dir, transform=transform)

# Create a DataLoader for your custom dataset
dataloader = DataLoader(custom_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

# Get the model
backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(device)
backbone.train()

# Main star of the show
margin_adding = ArcFace(s=64, margin=0.5)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD (
    backbone.parameters(),
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

wandb.watch (backbone, log_freq=100)

# To track training step for gradient accumulation
global_step: int = 0

for epoch in range(1, cfg.num_epochs + 1):
    print (f'Epoch No. {epoch} has started...')

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        local_embeddings = backbone(inputs)
        # loss: torch.Tensor = module_partial_fc(local_embeddings, labels)
        adjusted_margin_logits = margin_adding(local_embeddings, labels)

        loss = criterion(adjusted_margin_logits, labels)
        loss.backward()

        if global_step % cfg.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()

print ('Training Complete')