import tensorflow as tf
import numpy as np
import eval
import os
import fundus
import itertools



def ensemble_with_all_combibation(model_paths , test_images , test_labels):
    path , subdir_names , _=os.walk(model_paths).next()
    subdir_paths=map(lambda name : os.path.join(path , name) , subdir_names)
    print 'model saved folder paths : {}'.format(subdir_paths)
    max_acc=0
    for k in range(2,len(subdir_paths)):
        print 'K : {}'.format(k)
        for cbn_models in itertools.combinations(subdir_paths ,k):
            print cbn_models
            for idx ,cbn_model in enumerate(cbn_models):
                if idx ==0 :
                    pred_sum=eval.eval(cbn_model,test_images)
                else:
                    pred_sum+=eval.eval(cbn_model,test_images)
            print 'Combination Model {} '.format(cbn_models)
            pred_sum=pred_sum/float(len(cbn_models))
            acc=eval.get_acc(pred_sum , test_labels)
            if max_acc < acc :
                max_acc=acc
                max_list=cbn_models
            print 'max acc : {} , max_list {} '.format(max_acc,max_list)
    return acc , max_list




def ensemble(model_paths , test_images):
    """
    :param models:
    :return:

    """

    path , subdir_names , _=os.walk(model_paths).next()
    subdir_paths=map(lambda name : os.path.join(path , name) , subdir_names)
    print 'model saved folder paths : {}'.format(subdir_paths)

    for i,subdir_path in enumerate(subdir_paths):
        pred=eval.eval(subdir_path , test_images)
        if i ==0 :
            pred_sum = pred
        else:
            pred_sum+=pred
    pred_sum=pred_sum/float(i+1)
    return pred_sum

if __name__ == '__main__':
    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = fundus.type1(
        './fundus_300_debug', resize=(299, 299))

    acc, max_list=ensemble_with_all_combibation('./models' ,test_images , test_labels)
    """
    pred_sum=ensemble('./models', test_images )
    acc =eval.get_acc(pred_sum , test_labels)
    print acc
    """



