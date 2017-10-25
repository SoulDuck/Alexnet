# Alexnet
result 1 
conv_out_features=[16,16,16,16,16]

conv_kernel_sizes=[5,5,5,3,3]

conv_strides=[1,1,1,1,1]

before_act_bn_mode = []

after_act_bn_mode = []

allow_max_pool_indices=[0,1,4]

dropout cnn --> None 

fc_out_features = [1024,1024, n_classes]  

dropout fc layer = 0.5 

optimizer Momentum with nesterov

Loss = L2_loss * weight_decay(=0.9) + cross_entropy

![Alt_text](./readme_pic/acc_train.png)
![Alt_text](./readme_pic/loss_train.png)
![Alt_text](./readme_pic/acc_val.png)
![Alt_text](./readme_pic/loss_val.png)
