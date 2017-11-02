# Alexnet

## structure 

![Alt_text](readme_pic/structure.png)

## Fundus Classification 

conv_out_features=[32,64,64,64,128] | conv_kernel_sizes=[7,5,5,3,3] | conv_strides=[2,2,2,1,1]

allow_max_pool_indices=[0,1,4]

before_act_bn_mode = [] after_act_bn_mode = []

fc_out_features = [1024,1024]

Batch Size 80 

Data Normal 3000 | glaucoma 1000 , retina 1000 , cataract 1000 | Label : single Label 

|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | acc | loss | server | 
| --- | --- | --- | --- | --- | --- | --- |--- | --- | 
| SGD | X | X | X | FC | X |   ? | ? | ? |
| SGD | O | X | X | FC | X |   ? | ? | ? |
| SGD | O | O | X | FC | X |   ? | ? | ? |
| SGD | O | O | X | GAP| X |   ? | ? | ? |
| SGD | O | O | X | GAP | X |   ? | ? | ? |
| SGD | O | O | X | GAP | X |   ? | ? | ? |
| SGD | X | X | O | FC | X |   ? | ? | ? |
| SGD | O | X | O | FC | X |   ? | ? | ? |


|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | acc | loss | 
| --- | --- | --- | --- | --- | --- | --- |--- |
| SGD | O | O | X | FC | X | ? | ? | 
| SGD | O | O | X | GAP | X | ? | ? | 
| Momentum | O | O | X | FC | X | ? | ? | 
| Momentum | O | O | X | GAP | X | ? | ? | 
| Momentum+ | O | O | X | FC | X | ? | ? | 
| Momentum+ | O | O | X | GAP | X | ? | ? | 
| Adam | O | O | X | FC | X | ? | ? | 
| Adam | O | O | X | GAP | X | ? | ? | 
