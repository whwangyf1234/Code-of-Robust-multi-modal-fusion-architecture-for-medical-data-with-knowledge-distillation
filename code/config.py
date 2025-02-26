# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:10:00 2025

@author: user
"""

import torch
class Config:
    def __init__(self):
        
        
        
        #path name
        self.datalist_path = r'..//data//data_list_0525_4.csv'  #‘’    0624    0525_4
        self.bert_name = r"../../pretrained_model/microsoft_Pubmedbert"  #microsoft_Pubmedbert gatortron-base
        self.model_save_path = 'sequencelenth300'
        self.image_path = r'../data/image_tensor224.pt'  #image_all_tensor224.pt
        self.data_from = 'hf'  #hf heart failure

        #data_Dataset
        self.selftoken = None
        self.maxlen = None   #不知是啥
        self.padding = 'post'
        
        #train and test
        self.random_seed = 1
        self.rat = 0.8
        self.alpha_min = 0.
        self.alpha_max = 0.
        self.num_epochs = 10  #10 训练轮数
        self.num_epochs2 = 5  # 10 训练轮数
        self.num_epochs_pre = 1
        self.batch_size = 8  # 批大小
        self.step_print =100
        self.setp_stop = 200
        self.stay_time = 3       #6     3
        self.stay_time2 = 6
        self.rastay_time = 6  #1000  6
        self.rastay_time2 = 1000
        
        self.gamma = 0.6     #学习率衰竭系数
        self.learning_rate = 0.0001   #0.02
        self.tabnet_rate = 0.0001
        self.bert_rate = 0.00001
        self.img_rate = 0.0001
        
        self.weight_decay = 0.
        self.bert_decay = 0.
        self.img_decay = 0.
                
        self.num_classes = 2
        self.dropout = 0.
        
        self.clip_grad = False

        self.temperature = 1
        self.alpha = 1
        self.modulation_starts = 0
        self.modulation_ends = 3
        self.modulation = 'OGM_GE'  # OGM or OGM_GE

        self.p_weight = [0.1, 0.1, 0.8]  #蒸馏学习缺失样本概率 1/3, 1/3, 1/3   0.1, 0.1, 0.8

        #text
        self.freeze_bert = False
        self.sequence_size_text = 300
        self.dim_text = 768
        self.emb_dropout_text = 0.
        
        #image
        self.dim_img = 1024
        self.pool_size = 7
        self.img_lenth = self.pool_size * self.pool_size
        self.emb_dropout_img = 0.

        #tabular
        self.tabular_size = 140         #32  140
        self.dim_tabular = 512
        self.emb_dropout_tabular = 0.
        
        #tabnet
        self.tabnet_n_d = 512
        self.tabnet_n_a = self.tabnet_n_d
        self.tabnet_n_steps = 3
        
        #fusion
        self.fusion_lenth = 10
        self.dim_fusion = 512
        self.pool_fusion = 'weight'   # mean  or weight     weight是加权平均
        #attenion
        self.depth = 2
        self.heads = 2
        self.dim_feedforward = 512
        self.activation = 'gelu'  # relu or gelu or a unary callable
        self.pool = 'cls'  #mean
        self.layer_norm_eps = 1e-5  #default=1e-5
        self.batch_first = True
        self.norm_first = True   #True or False


        self.device = torch.device('cuda:0')   #torch.device('cpu')