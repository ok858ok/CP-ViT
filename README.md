# CP-ViT
Code for "CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction" (CVPR 2022) on CIFAR-10/100.

CP-ViT: a cascade pruning framework named CP-ViT by predicting sparsity in ViT models progressively and dynamically to reduce computational redundancy while minimizing the accuracy loss. Specifically, we define the cumulative score to reserve the informative patches and heads across the ViT model for better accuracy. We also propose the dynamic pruning ratio adjustment technique based on layer-aware attention range. CP-ViT has great general applicability for practical deployment, which can be applied to a wide range of ViT models and can achieve superior accuracy with or without finetuning.

## Usage
### Prerequisite
We have tested our codes under the following environments:
```
python == 3.9.5
pytorch == 1.9.0
CUDA == 11.2
```
### Pretrained Vision Transformer Models
To start with, you can first download pre-trained models from:

[ViT-B_16/224 cifar-10](https://pan.baidu.com/s/1NN4k05BWpUw2tHuqjDjY8g)

[ViT-B_16/224 cifar-100](https://pan.baidu.com/s/1XVY62ik2pptQvqspnIxmuA)

and place them under folder```./CP-ViT/output/```.
Of course you can download other pre-trained models from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) .

### Pruning without Finetuning
We then prune ViT model without finetuning by:
```
python3 eval.py \
        --name="CP-ViT test" \
        --dataset="cifar10" \
        --model_type="ViT-B_16" \
        --pretrained_dir='output/cifar10_checkpoint.pth' \
        --eval_batch_size=64 
```
### Pruning with Finetuning
We can finetune the CP-ViT model by:
```
python3 train.py \
        --name="CP-ViT finetune" \
        --dataset="cifar10" \
        --model_type="ViT-B_16" \
        --pretrained_dir='output/cifar10_checkpoint.pth' \
        --train_batch_size=64 \
        --eval_every=3125 \
        --learning_rate=3e-2 \
        --num_steps=10000 \
        --decay_type="cosine" 
```
### Acknowledge Related Repos
Pytorch Image Models: [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

ViT: [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

DeiT: [https://github.com/facebookresearch/deit](https://github.com/facebookresearch/deit)
