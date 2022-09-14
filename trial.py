

import os, random
import numpy as np
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10

import torchvision.transforms as T

import nni
import timm
from nni.algorithms.compression.v2.pytorch import TorchEvaluator
from nni.compression.pytorch.pruning import MovementPruner

from typing import Optional, Callable, Callable, Dict

def remain_weights(model, dim: int = 0):
    total, remain = 0, 0
    for wrapper in model.get_modules_wrapper().values():
        weight_mask = wrapper.weight_mask
        mask_size = weight_mask.size()
        if len(mask_size) == 1:
            index = torch.nonzero(weight_mask.abs() != 0, as_tuple=False).tolist()
        else:
            sum_idx = list(range(len(mask_size)))
            sum_idx.remove(dim)
            index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0, as_tuple=False).tolist()
        total += weight_mask.size(dim)
        remain += len(index)
    return remain/total


def training(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss, 
    lr_scheduler: optim.lr_scheduler._LRScheduler = None,
    max_steps: int = None,
    max_epochs: int = None,
    dataloader: DataLoader = None,
    evaluation_func: Callable = None,
    eval_per_steps: int = 200,
    device=None
):
    assert dataloader is not None

    model.train()
    current_step = 0

    total_epochs = max_steps // len(dataloader) + 1 if max_steps else max_epochs if max_epochs else 3
    total_steps = max_steps if max_steps else total_epochs * len(dataloader)

    print(f'Training {total_epochs} epochs, {total_steps} steps...')

    for current_epoch in range(total_epochs):
        for inputs, targets in dataloader:
            if current_step >= total_steps:
                return
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # per step schedule
            if lr_scheduler is not None:
                lr_scheduler.step()

            current_step += 1

            if current_step % eval_per_steps == 0:
                acc = evaluation_func(model) if evaluation_func else None
                nni.report_intermediate_result({'default': acc, 'sparsity': 1.0})
                print(f'Epoch {current_epoch:>2d}, Step {current_step:>5d}: {acc:.2%}')


@torch.no_grad()
def evaluation(model: torch.nn.Module, dataloader: Dict[str, DataLoader] = None, device=None):
    assert dataloader is not None
    training = model.training
    model.eval()

    total, correct = 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total += targets.size(0)
        correct += (outputs.argmax(dim=-1) == targets).sum().item()

    model.train(training)
    return correct / total


def create_finetuned_model(device=None):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.reset_classifier(args.num_classes, global_pool=None)
    model.load_state_dict(torch.load(args.weight))
    if device is not None:
        model = model.to(device)
    return model
        
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_num = 12
    patch_size = 16
    embed_dim = 192
    heads_num = 3

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

    dataloader_config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': False
    }
    train_loader = DataLoader(train_set, shuffle=True, **dataloader_config)
    valid_loader = DataLoader(valid_set, shuffle=False, **dataloader_config)

    steps_per_epoch = len(train_loader)
    # Set training steps/epochs for pruning.
    if not args.dev:
        total_epochs = args.epochs
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = 0
        cooldown_steps = args.epochs//10 * steps_per_epoch
    else:
        total_epochs = 2
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = 0
        cooldown_steps = 1

    # Initialize evaluator used by MovementPruner.

    def lr_lambda(current_step: int):
        return float(current_step) / warmup_steps if current_step < warmup_steps else max(0.001, float(total_steps - current_step) / float(total_steps - warmup_steps))

    model = create_finetuned_model(device)
    model.cls_token.requires_grad = False
    model.pos_embed.requires_grad = False
    model.patch_embed.requires_grad = False
    
    # prune the attention layer with MovementPruner.
    evaluation_func = partial(evaluation, dataloader=valid_loader, device=device)
    movement_training = partial(training, dataloader=train_loader, evaluation_func=evaluation_func, device=device)
    traced_optimizer = nni.trace(optim.Adam)(model.parameters(), lr=args.learning_rate, eps=1e-8)
    traced_scheduler = nni.trace(optim.lr_scheduler.LambdaLR)(traced_optimizer, lr_lambda)
    evaluator = TorchEvaluator(movement_training, traced_optimizer, nn.CrossEntropyLoss(), traced_scheduler)

    config_list = [{
        'op_types': ['Linear'],
        'op_partial_names': [f'blocks.{i}{args.name}' for i in range(layers_num)],
        'sparsity': args.sparsity
    }]

    pruner = MovementPruner(
        model=model,
        config_list=config_list,
        evaluator=evaluator,
        training_epochs=total_epochs,
        training_steps=total_steps,
        warm_up_step=warmup_steps,
        cool_down_beginning_step=total_steps - cooldown_steps,
        regular_scale=args.regular_scale,
        movement_mode='soft',
        sparse_granularity='auto'
    )
    _, attention_masks = pruner.compress()
    
    folder = f'experiment/{nni.get_trial_id()}'
    os.makedirs(folder, exist_ok=True)
    torch.save(attention_masks, os.path.join('experiment', f'{nni.get_trial_id()}.pth'))

    acc = evaluation_func(model)
    sparsity = 1.0 - remain_weights(pruner)
    nni.report_final_result({'default': round(acc, 4), 'sparsity': round(sparsity, 4)})



if __name__ == "__main__":
    HyperParameter = {
        'sparsity': 0.5,
        'regular_scale': 2,
        'name': 'attn',
    }
    HyperParameter.update(nni.get_next_parameter())

    class args:
        name = HyperParameter["name"]
        sparsity = HyperParameter["sparsity"]
        regular_scale = HyperParameter["regular_scale"]
        datapath = "/root/rjchen/data"
        num_classes = 10
        weight = "finetuned.pth"
        seed = 42
        num_workers = 4
        batch_size = 128
        epochs = 30
        learning_rate = 0.0001
        dev = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)