# CP-ViT
Code for "CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction" (CVPR 2022) on CIFAR-10/100.

CP-ViT: a cascade pruning framework named CP-ViT by predicting sparsity in ViT models progressively and dynamically to reduce computational redundancy while minimizing the accuracy loss. Specifically, we define the cumulative score to reserve the informative patches and heads across the ViT model for better accuracy. We also propose the dynamic pruning ratio adjustment technique based on layer-aware attention range. CP-ViT has great general applicability for practical deployment, which can be applied to a wide range of ViT models and can achieve superior accuracy with or without finetuning.
