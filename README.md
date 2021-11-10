# CP-ViT
Code for "CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction" (CVPR 2022) on CIFAR-10/100.

CP-ViT: a cascade pruning framework named CP-ViT by predicting sparsity in ViT models progressively and dynamically to reduce computational redundancy while minimizing the accuracy loss. Specifically, we define the cumulative score to reserve the informative patches and heads across the ViT model for better accuracy. We also propose the dynamic pruning ratio adjustment technique based on layer-aware attention range. CP-ViT has great general applicability for practical deployment, which can be applied to a wide range of ViT models and can achieve superior accuracy with or without finetuning.

### Usage
#### Prerequisite
We have tested our codes under the following environments:
```
python == 3.9.5
pytorch == 1.9.0
CUDA == 11.2
```
#### Pretrained Vision Transformer Models
To start with, you can first download a pre-trained model from:
[ViT-B_16/224 cifar-10](https://pan.baidu.com/s/1NN4k05BWpUw2tHuqjDjY8g)
[ViT-B_16/224 cifar-100](https://pan.baidu.com/s/1XVY62ik2pptQvqspnIxmuA)
and place them under folder```./CP-ViT/output/```.
#### Pruning Without Finetuning
We then prune ViT model without finetuning by:
```
python3 eval.py \
        --name="CP-ViT test" \
        --dataset="cifar10" \
        --model_type="ViT-B_16" \
        --pretrained_dir='output/cifar10_checkpoint.pth' \
        --eval_batch_size=64 \
```
