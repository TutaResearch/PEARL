import os
import random
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

from models.model import PEARL, PEARLFeature
from models.loss import medical_codes_loss
from metrics import EvaluateCodesCallBack, EvaluateHFCallBack
from utils import DataGenerator, lr_decay
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def datacluster(data,num):
    model = AgglomerativeClustering(n_clusters=int(num))
    yhat = model.fit_predict(data)
    return yhat
    

def load_data(dataset_path):
    encoded_path = os.path.join(dataset_path, 'encoded')
    standard_path = os.path.join(dataset_path, 'standard')
    code_maps = pickle.load(open(os.path.join(encoded_path, 'code_maps.pkl'), 'rb'))
    pretrain_codes_data = pickle.load(open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'rb'))
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    hf_dataset = pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return code_maps, pretrain_codes_data, codes_dataset, hf_dataset, auxiliary


if __name__ == '__main__':
    dataset = 'mimic3'
    dataset_path = os.path.join('data', dataset)
    code_maps, pretrain_codes_data, codes_dataset, hf_dataset, auxiliary = load_data(dataset_path)
    code_map, code_map_pretrain = code_maps['code_map'], code_maps['code_map_pretrain']
    (train_codes_data, valid_codes_data, test_codes_data) = (codes_dataset['train_codes_data'],
                                                             codes_dataset['valid_codes_data'],
                                                             codes_dataset['test_codes_data'])
    (train_hf_y, valid_hf_y, test_hf_y) = hf_dataset['train_hf_y'], hf_dataset['valid_hf_y'], hf_dataset['test_hf_y']

    (pretrain_codes_x, pretrain_codes_y, pretrain_y_h, pretrain_visit_lens) = pretrain_codes_data
    (train_codes_x, train_codes_y, train_y_h, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_y_h, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_y_h, test_visit_lens) = test_codes_data
    (code_levels, code_levels_pretrain,
     subclass_maps, subclass_maps_pretrain,
     code_code_adj) = (auxiliary['code_levels'], auxiliary['code_levels_pretrain'],
                       auxiliary['subclass_maps'], auxiliary['subclass_maps_pretrain'],
                       auxiliary['code_code_adj'])

    op_conf = {
        'pretrain': True,
        'from_pretrain': False,
        'pretrain_path': './saved/hyperbolic/%s/PEARL_a/PEARL_pretrain' % dataset,
        'use_embedding_init': True,
        'use_hierarchical_decoder': True,
        'task': 'm',  # m: medical codes, h: heart failure
    }

    feature_model_conf = {
        'code_num': len(code_map_pretrain),
        'code_embedding_init': None,
        'adj': code_code_adj,
        'max_visit_num': train_codes_x.shape[1]
    }

    pretrain_model_conf = {
        'use_hierarchical_decoder': op_conf['use_hierarchical_decoder'],
        'subclass_dims': np.max(code_levels_pretrain, axis=0) if op_conf['use_hierarchical_decoder'] else None,
        'subclass_maps': subclass_maps_pretrain if op_conf['use_hierarchical_decoder'] else None,
        'output_dim': len(code_map_pretrain),
        'activation': None
    }

    task_conf = {
        'm': {
            'output_dim': len(code_map),
            'activation': None,
            'loss_fn': medical_codes_loss,
            'label': {
                'train': train_codes_y.astype(np.float32),
                'valid': valid_codes_y.astype(np.float32),
                'test': test_codes_y.astype(np.float32)
            },
            'evaluate_fn': EvaluateCodesCallBack
        },
        'h': {
            'output_dim': 1,
            'activation': 'sigmoid',
            'loss_fn': 'binary_crossentropy',
            'label': {
                'train': train_hf_y.astype(np.float32),
                'valid': valid_hf_y.astype(np.float32),
                'test': test_hf_y.astype(np.float32)
            },
            'evaluate_fn': EvaluateHFCallBack
        }
    }

    model_conf = {
        'use_hierarchical_decoder': False,
        'output_dim': task_conf[op_conf['task']]['output_dim'],
        'activation': task_conf[op_conf['task']]['activation']
    }

    hyper_params = {
        'clients_num':4,
        'code_embedding_size': 128,
        'hiddens': [64],
        'attention_size_code': 64,
        'attention_size_visit': 32,
        'patient_size': 64,
        'patient_activation': tf.keras.layers.LeakyReLU(),
        'pretrain_epoch': 5, #local pretrain epoch
        'exchange_epoch': 100, #federated pretrain epoch
        'pretrain_batch_size': 128,
        'epoch': 200, #fine-tuning
        'batch_size': 32,
        'gnn_dropout_rate': 0.8,
        'decoder_dropout_rate': 0.17
    }

    if op_conf['use_embedding_init']:
        if op_conf['pretrain'] or (not op_conf['from_pretrain']):
            embedding_init = pickle.load(open('./saved/hyperbolic/%s_leaf_embeddings' % dataset, 'rb'))
            feature_model_conf['code_embedding_init'] = embedding_init
    PEARL_feature = PEARLFeature(feature_model_conf, hyper_params)
    #print(embedding_init.shape)
    #print(PEARL_feature.shape)

    # pretrain
    if op_conf['pretrain']:
        pretrain_x_tol=[]
        pretrain_y_tol=[]
        y_index=datacluster(np.vstack((pretrain_y_h[0],test_y_h[0])),int(hyper_params['clients_num']))
        print(len(pretrain_y_h[0]))

        index = [[] for i in range(int(hyper_params['clients_num']))]
        for i, ind in enumerate(y_index[:-1000]):
            index[ind].append(i)
        index_y=[np.array(index[i], dtype = int) for i in range(int(hyper_params['clients_num']))]


        for i in range(int(hyper_params['clients_num'])):
            pretrain_x = {
                'visit_codes': np.array([pretrain_codes_x[j] for j in index_y[i]]),
                'visit_lens': np.array([pretrain_visit_lens[j] for j in index_y[i]])
            }
      
            if op_conf['use_hierarchical_decoder']:
                pretrain_x['y_trues'] = [np.array([pretrain_y_h[0][j] for j in index_y[i]]),np.array([pretrain_y_h[1][j] for j in index_y[i]]),np.array([pretrain_y_h[2][j] for j in index_y[i]]),np.array([pretrain_y_h[3][j] for j in index_y[i]])]
                #print(np.array(pretrain_y_h[0]))
 
                pretrain_y = None
            else:
                pretrain_y = np.array([pretrain_codes_y[j] for j in index_y[i]]).astype(np.float32)
            pretrain_x_tol.append(pretrain_x)
            pretrain_y_tol.append(pretrain_y)

            
        init_lr = 1e-2
        # split_val = [(20, 1e-3), (150, 1e-4), (500, 1e-5)]
        # split_val = [(100, 1e-3), (500, 1e-4)]
        split_val = [(3, 1e-2)]
        lr_schedule_fn = lr_decay(total_epoch=hyper_params['pretrain_epoch'], init_lr=init_lr, split_val=split_val)#epoch
        lr_scheduler = LearningRateScheduler(lr_schedule_fn)

        loss_fn = None if op_conf['use_hierarchical_decoder'] else medical_codes_loss
        PEARL_pretrain = PEARL(PEARL_feature, pretrain_model_conf, hyper_params)
        PEARL_pretrain.compile(optimizer='rmsprop', loss=loss_fn)
        for e in range(int(hyper_params['exchange_epoch'])):
            if e==int(100/int(hyper_params['pretrain_epoch'])):
                init_lr = 1e-3
                split_val = [(2, 1e-3)]
                lr_schedule_fn = lr_decay(total_epoch=hyper_params['pretrain_epoch'], init_lr=init_lr, split_val=split_val)#epoch
                lr_scheduler = LearningRateScheduler(lr_schedule_fn)
            if e==int(250/int(hyper_params['pretrain_epoch'])):
                init_lr = 1e-4
                split_val = [(2, 1e-4)]
                lr_schedule_fn = lr_decay(total_epoch=hyper_params['pretrain_epoch'], init_lr=init_lr, split_val=split_val)#epoch
                lr_scheduler = LearningRateScheduler(lr_schedule_fn)
            if e==int(400/int(hyper_params['pretrain_epoch'])):
                init_lr = 1e-5
                split_val = [(2, 1e-5)]
                lr_schedule_fn = lr_decay(total_epoch=hyper_params['pretrain_epoch'], init_lr=init_lr, split_val=split_val)#epoch
                lr_scheduler = LearningRateScheduler(lr_schedule_fn)
            print('epoch.............................')
            print(e)

            '''
            # locolPEARL
            for i in range(0,1):
                PEARL_pretrain.fit(x=pretrain_x_tol[i], y=pretrain_y_tol[i],
                                   batch_size=hyper_params['pretrain_batch_size'], epochs=hyper_params['pretrain_epoch'],
                                   callbacks=[lr_scheduler])
            '''


            wei=[]
            for i in range(int(hyper_params['clients_num'])):
                if e != 0:
                    PEARL_pretrain.set_weights(weight_aggre)
                
                PEARL_pretrain.fit(x=pretrain_x_tol[i], y=pretrain_y_tol[i],
                                   batch_size=hyper_params['pretrain_batch_size'], epochs=hyper_params['pretrain_epoch'],initial_epoch=0,
                                   callbacks=[lr_scheduler])
                wei.append(PEARL_pretrain.get_weights())
                
            weight_aggre=wei[0].copy()
            for i in range(len(wei[0])):
                for j in range(1,int(hyper_params['clients_num'])):
                    weight_aggre[i]+=(wei[j][i]+np.random.laplace(0, init_lr/1, len(wei[j][i])*len(wei[j][i][0])).reshape(len(wei[j][i]),len(wei[j][i][0]))) if len(wei[j][i].shape)==2 else (wei[j][i]+np.random.laplace(0, init_lr/1, len(wei[j][i])))
                weight_aggre[i]=weight_aggre[i]/int(hyper_params['clients_num'])
            

        #PEARL_pretrain.set_weights(weight_aggre)
        PEARL_pretrain.set_weights(weight_aggre)
        
        PEARL_pretrain.save_weights(op_conf['pretrain_path'])
        
    
    
    # fine-tuning
    else:
        if op_conf['from_pretrain']:
            PEARL_pretrain = PEARL(PEARL_feature, pretrain_model_conf, hyper_params)
            PEARL_pretrain.load_weights(op_conf['pretrain_path'])
        x_tol=[]
        valid_x_tol=[]
        y_tol=[]
        valid_y_tol=[]

        y_index=datacluster(np.vstack((pretrain_y_h[0],test_y_h[0])),int(hyper_params['clients_num']))
        print(y_index)

        index = [[] for i in range(int(hyper_params['clients_num']))]
        for i, ind in enumerate(y_index[-7000:-1000]):
            index[ind].append(i)
        index_y=[np.array(index[i], dtype = int) for i in range(int(hyper_params['clients_num']))]

        index_test = [[] for i in range(int(hyper_params['clients_num']))]
        for i, ind in enumerate(y_index[-1000:]):
            index_test[ind].append(i)
        index_y_test=[np.array(index_test[i], dtype = int) for i in range(int(hyper_params['clients_num']))]

        #np.array([valid_visit_lens[j] for j in index_y_test[i]])

        for i in range(int(hyper_params['clients_num'])):
            x = {
                'visit_codes': np.array([train_codes_x[j] for j in index_y[i]]),
                'visit_lens': np.array([train_visit_lens[j] for j in index_y[i]])
            }
            valid_x = {
                'visit_codes': valid_codes_x[i::int(hyper_params['clients_num'])],
                'visit_lens': valid_visit_lens[i::int(hyper_params['clients_num'])]
            }
            y = np.array([task_conf[op_conf['task']]['label']['train'][j] for j in index_y[i]])
            valid_y = task_conf[op_conf['task']]['label']['valid']
            test_y = task_conf[op_conf['task']]['label']['test']#[i::int(hyper_params['clients_num'])]
            x_tol.append(x)
            valid_x_tol.append(valid_x)
            y_tol.append(y)
            valid_y_tol.append(valid_y)

        init_lr = 1e-2
        split_val = [(20, 1e-3), (35, 1e-4), (100, 1e-5)]

        lr_schedule_fn = lr_decay(total_epoch=hyper_params['epoch'], init_lr=init_lr, split_val=split_val)
        

        # IID
        for i in range(3,4):
            test_codes_gen = DataGenerator([np.array([test_codes_x[j] for j in index_y_test[i]]), np.array([test_visit_lens[j] for j in index_y_test[i]])], shuffle=False, batch_size=128)

            loss_fn = task_conf[op_conf['task']]['loss_fn']
            lr_scheduler = LearningRateScheduler(lr_schedule_fn)
            test_callback = task_conf[op_conf['task']]['evaluate_fn'](test_codes_gen, np.array([test_y[j] for j in index_y_test[i]]))
            PEARL = PEARL(PEARL_feature, model_conf, hyper_params)
            PEARL.compile(optimizer='rmsprop', loss=loss_fn)
        
            PEARL.fit(x=x_tol[i], y=y_tol[i],
                            batch_size=hyper_params['batch_size'], epochs=hyper_params['epoch'],
                            callbacks=[lr_scheduler, test_callback])
            PEARL.summary()



        '''
        # Non-IID
        test_codes_gen = DataGenerator([test_codes_x, test_visit_lens], shuffle=False, batch_size=128)

        loss_fn = task_conf[op_conf['task']]['loss_fn']
        lr_scheduler = LearningRateScheduler(lr_schedule_fn)
        test_callback = task_conf[op_conf['task']]['evaluate_fn'](test_codes_gen, test_y)
        PEARL = PEARL(PEARL_feature, model_conf, hyper_params)
        PEARL.compile(optimizer='rmsprop', loss=loss_fn)
        #for i in range(int(hyper_params['clients_num'])):
        for i in range(3,4):
            PEARL.fit(x=x_tol[i], y=y_tol[i],
                            batch_size=hyper_params['batch_size'], epochs=hyper_params['epoch'],
                            callbacks=[lr_scheduler, test_callback])
            PEARL.summary()
            #tf.reset_default_graph()
        '''