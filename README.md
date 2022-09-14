
# NNI Movement Pruning ViT 

## Usage

### Finetune ImageNet-1k pretrained model to CIFAR10
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --epochs 50 --warmup 5 --gpus 4 --learning_rate 0.0005
```

### Grid Search performance
```
pytho tuner.py
```