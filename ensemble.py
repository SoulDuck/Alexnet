import tensorflow as tf
import numpy as np
import eval
import os
import fundus
import itertools
import pickle

def get_models_paths(dir_path):
    subdir_paths=[path[0] for path in os.walk(dir_path)]
    #subdir_paths = map(lambda name: os.path.join(dir_path, name), subdir_names)
    ret_subdir_paths=[]
    for path in subdir_paths:
        if os.path.isfile(os.path.join(path , 'model.meta')):
            ret_subdir_paths.append(path)
    return ret_subdir_paths


def ensemble_with_all_combibation(model_paths , test_images , test_labels):
    max_acc=0
    f = open('best_ensemble.txt', 'w')

    if not os.path.isfile('predcitions.pkl'):
        p = open('predcitions.pkl' , 'w')
        pred_dic={}
        for path in model_paths:
            try:
                pred_sum = eval.eval(path, test_images)
            except:
                print 'Error Path ',path
                exit()
            pred_dic[path]=pred_sum
        #pred_model_path_list=zip(pred_list , model_paths)
        pickle.dump(pred_dic,p)

    else:
        p = open('predcitions.pkl', 'r')
        pred_dic=pickle.load(p)
    p.close()

    for k in range(2,len(model_paths)+1):
        k_max_acc = 0
        k_max_list = []
        print 'K : {}'.format(k)
        for cbn_models in itertools.combinations(pred_dic.keys(),k):
            cbn_preds=map(lambda cbn_model: pred_dic[cbn_model],cbn_models)
            for idx ,pred in enumerate(cbn_preds):
                if idx ==0 :
                    pred_sum=pred
                else:
                    pred_sum+=pred

            pred_sum = pred_sum / float(len(cbn_models))
            acc=eval.get_acc(pred_sum , test_labels)
            print acc
            if max_acc < acc :
                max_acc=acc
                max_list=cbn_models
            if k_max_acc < acc:
                k_max_acc = acc
                k_max_list = cbn_models
        msg = 'k : {} , list : {} , accuracy : {}\n'.format(k, k_max_list , k_max_acc)
        f.write(msg)
        f.flush()
    msg='model list : {} , accuracy : {}'.format(max_list , max_acc)
    f.write(msg)
    f.flush()

    return acc , max_list

"""
def ensemble_with_all_combibation(model_paths , test_images , test_labels):
    max_acc=0
    f = open('best_ensemble.txt', 'w')
    for k in range(2,len(model_paths)+1):
        k_max_acc = 0
        k_max_list = []
        print 'K : {}'.format(k)
        for cbn_models in itertools.combinations(model_paths ,k):
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
            if k_max_acc < acc:
                k_max_acc = acc
                k_max_list = cbn_models
        msg = 'k : {} , list : {} , accuracy : {}\n'.format(k, max_list , max_acc)
        f.write(msg)
        f.flush()
    msg='model list : {} , accuracy : {}'.format(max_list , max_acc)
    f.write(msg)
    f.flush()

    return acc , max_list
"""




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
    model_paths=get_models_paths('./models')
    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = fundus.type1(
        './fundus_300', resize=(299, 299))
    acc, max_list=ensemble_with_all_combibation(model_paths ,test_images , test_labels)

    """
    pred_sum=ensemble('./models', test_images )
    acc =eval.get_acc(pred_sum , test_labels)
    print acc
    """



