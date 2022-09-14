import warnings
warnings.filterwarnings("ignore")

import os, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, ImageFolder, ImageNet

import torchvision.transforms as T


from tqdm import tqdm
import torch
import timm


def train(epoch, model, dataloader, criterion, optimizer, device):
    
    Loss, Acc = 0, 0
    batches = 0
    model.train()
    with tqdm(dataloader, total=len(dataloader), desc=f"Train({epoch})", ncols=100) as t:
        for inputs, targets in t:
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batches += 1
            Loss += loss.item()
            Acc += (outputs.argmax(dim=-1) == targets).sum().item() / targets.size(0)

            t.set_postfix({"Loss": f"{Loss/batches:.3f}", "Acc": f"{Acc/batches:.2%}"})
    return Loss/batches, Acc/batches

@torch.no_grad()
def valid(epoch, model, dataloader, device):
    model.eval()
    top1, top5, total = 0, 0, 0
    with tqdm(dataloader, total=len(dataloader), desc=f"Valid({epoch})", ncols=100) as t:
        for inputs, targets in t:
            inputs, targets = inputs.to(device), targets.to(device)
            topk = torch.topk(model(inputs), dim=-1, k=5, largest=True, sorted=True).indices
            correct = topk.eq(targets.view(-1, 1).expand_as(topk))
            top1 += correct[:, 0].sum().item()
            top5 += correct[:, :5].sum().item()
            total += targets.size(0)
            t.set_postfix({"Top1": f"{top1/total:.2%}", "Top5": f"{top5/total:.2%}"})
    return top1/total, top5/total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

    train_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    train_set = CIFAR10(args.datapath, download=False, train=True, transform=train_transform)
    valid_set = CIFAR10(args.datapath, download=False, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # transform = T.Compose([
    #     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ])

    # dataset = ImageFolder("../../data/imagenet/val", transform=transform)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # valid(-1, model, dataloader, device)
    
    model.reset_classifier(args.num_classes, global_pool=None)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train(epoch, model, train_loader, criterion, optimizer, device)
        valid(epoch, model, valid_loader, device)
    
    torch.save(model.state_dict(), "finetuned.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("ViT fine-tune")
    parser.add_argument("--datapath", type=str, default="/root/rjchen/data")
    parser.add_argument("--num_classes", type=int, default=10)
    # parser.add_argument("--weights", type=str, required=True, default="")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    args = parser.parse_args()

    main(args)