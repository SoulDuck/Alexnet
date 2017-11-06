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

|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | acc | loss | 
| --- | --- | --- | --- | --- | --- | --- |--- |
| SGD | X | X | X | FC | X |   80.1% | 0.455 |
| SGD | O | X | X | FC | X |   81.07 | 0.46 |
| SGD | O | O | X | FC | X |   81.25 | 0.43 |
| SGD | O | O | X | GAP| X |   80.08 | 0.44 |
| SGD | O | O | X | GAP | X |  80.00 | 0.46 |
| SGD | O | O | X | GAP | X |  80.08 | 0.47 |
| SGD | X | X | O | FC | X |   82.14 | 0.44 |
| SGD | O | X | O | FC | X |   82.50 | 0.44 |


|Optimizer| augmentation | random crop | L2 loss | Fc or gap | batch norm | acc | loss | 
| --- | --- | --- | --- | --- | --- | --- |--- |
| SGD | O | O | X | FC | X | 79.46 | 0.47 | 
| SGD | O | O | X | GAP | X | 81.07 | 0.43 | 
| Momentum+ | O | O | X | FC | X | 78.57 | 0.47 | 
| Momentum+ | O | O | X | GAP | X | ? | ? | 
| Adam | O | O | X | FC | X | ? | ? | 
| Adam | O | O | X | GAP | X | ? | ? | 



| Momentum+ | O | O | X | FC | X | 78.57 | 0.47 |
학습이 안된 이유는 epoch을 너무 적게 잡았다.

| Adam | O | O | X | FC | X | ? | ? | 
learning rate 을 조정해야 햔다 .
