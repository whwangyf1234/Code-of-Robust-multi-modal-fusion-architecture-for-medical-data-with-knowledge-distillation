# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:57:14 2025

@author: user
"""


import torchvision.models as models
from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from pytorch_tabnet.tab_network import TabNet
from pytorch_tabnet.utils import create_group_matrix
class Transformer_fusion_missing(nn.Module):  #与缺失模态token做池化
    def __init__(self, fusion_lenth, dim_fusion, depth, heads, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first , pool_fusion):
        super().__init__()
        self.fusion_lenth = fusion_lenth
        self.pool = pool_fusion
        self.fusion_token_img = nn.Parameter(torch.randn(1, fusion_lenth, dim_fusion))
        self.fusion_token_text = nn.Parameter(torch.randn(1, fusion_lenth, dim_fusion))
        self.fusion_token_tabular = nn.Parameter(torch.randn(1, fusion_lenth, dim_fusion))
        self.layers_text = nn.ModuleList([])
        self.layers_img = nn.ModuleList([])
        self.layers_tabular = nn.ModuleList([])
        if pool_fusion == 'weight':
            self.mean_weights = nn.Parameter(torch.ones(3))
        if pool_fusion == 'mean':
            self.mean_weights = nn.Parameter(torch.ones(3))
            self.mean_weights.requires_grad = False

        for _ in range(depth):
            self.layers_text.append(nn.TransformerEncoderLayer(dim_fusion, heads, dim_feedforward, dropout, activation, layer_norm_eps,batch_first, norm_first))
            self.layers_img.append(nn.TransformerEncoderLayer(dim_fusion, heads, dim_feedforward, dropout, activation, layer_norm_eps,batch_first, norm_first))
            self.layers_tabular.append(nn.TransformerEncoderLayer(dim_fusion, heads, dim_feedforward, dropout, activation, layer_norm_eps,batch_first, norm_first))

    def forward(self, img, text, tabular):
        missing_modalities = [img, text, tabular].count(None)

        if missing_modalities == 0:
            b, c_text, _ = text.shape
            fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
            text = torch.cat((fusion_token_text, text), dim=1)
            fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
            img = torch.cat((fusion_token_img, img), dim=1)
            fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
            tabular = torch.cat((fusion_token_tabular, tabular), dim=1)
            for trm_text, trm_img, trm_tabular in zip(self.layers_text, self.layers_img, self.layers_tabular):
                text = trm_text(text)
                fusion_token_text = text[:, :self.fusion_lenth]
                img = trm_img(img)
                fusion_token_img = img[:, :self.fusion_lenth]
                tabular = trm_tabular(tabular)
                fusion_token_tabular = tabular[:, :self.fusion_lenth]
                fusion_token = torch.sum(torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0)/torch.sum(self.mean_weights)
                text[:, :self.fusion_lenth] = fusion_token
                img[:, :self.fusion_lenth] = fusion_token
                tabular[:, :self.fusion_lenth] = fusion_token

        elif missing_modalities == 1:
            if img is None:
                b, c_text, _ = text.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                text = torch.cat((fusion_token_text, text), dim=1)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                tabular = torch.cat((fusion_token_tabular, tabular), dim=1)
                for trm_text, _, trm_tabular in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    text = trm_text(text)
                    fusion_token_text = text[:, :self.fusion_lenth]
                    tabular = trm_tabular(tabular)
                    fusion_token_tabular = tabular[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    text[:, :self.fusion_lenth] = fusion_token
                    tabular[:, :self.fusion_lenth] = fusion_token
            elif text is None:
                b, c_text, _ = img.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                img = torch.cat((fusion_token_img, img), dim=1)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                tabular = torch.cat((fusion_token_tabular, tabular), dim=1)
                for _, trm_img, trm_tabular in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    img = trm_img(img)
                    fusion_token_img = img[:, :self.fusion_lenth]
                    tabular = trm_tabular(tabular)
                    fusion_token_tabular = tabular[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    img[:, :self.fusion_lenth] = fusion_token
                    tabular[:, :self.fusion_lenth] = fusion_token
            elif tabular is None:
                b, c_text, _ = text.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                text = torch.cat((fusion_token_text, text), dim=1)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                img = torch.cat((fusion_token_img, img), dim=1)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                for trm_text, trm_img, _ in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    text = trm_text(text)
                    fusion_token_text = text[:, :self.fusion_lenth]
                    img = trm_img(img)
                    fusion_token_img = img[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    text[:, :self.fusion_lenth] = fusion_token
                    img[:, :self.fusion_lenth] = fusion_token

        elif missing_modalities == 2:
            if text is not None:
                b, c_text, _ = text.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                text = torch.cat((fusion_token_text, text), dim=1)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                for trm_text, _, _ in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    text = trm_text(text)
                    fusion_token_text = text[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    text[:, :self.fusion_lenth] = fusion_token
            elif tabular is not None:
                b, c_text, _ = tabular.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                tabular = torch.cat((fusion_token_tabular, tabular), dim=1)
                for _, _, trm_tabular in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    tabular = trm_tabular(tabular)
                    fusion_token_tabular = tabular[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    tabular[:, :self.fusion_lenth] = fusion_token
            elif img is not None:
                b, c_text, _ = img.shape
                fusion_token_text = repeat(self.fusion_token_text, '() n d -> b n d', b=b)
                fusion_token_img = repeat(self.fusion_token_img, '() n d -> b n d', b=b)
                img = torch.cat((fusion_token_img, img), dim=1)
                fusion_token_tabular = repeat(self.fusion_token_tabular, '() n d -> b n d', b=b)
                for _, trm_img, _ in zip(self.layers_text, self.layers_img, self.layers_tabular):
                    img = trm_img(img)
                    fusion_token_img = img[:, :self.fusion_lenth]
                    fusion_token = torch.sum(
                        torch.stack([fusion_token_text, fusion_token_img, fusion_token_tabular]) * (
                            self.mean_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=0) / torch.sum(
                        self.mean_weights)
                    img[:, :self.fusion_lenth] = fusion_token
        return fusion_token
    
    
    
class TEXT_IMG_Tabular_SA1(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.config = config
        config_bert = AutoConfig.from_pretrained(config.bert_name)
        config_bert.update({'attention_probs_dropout_prob': config.dropout, 'hidden_dropout_prob': config.dropout})
        self.bert_text = AutoModel.from_pretrained(config.bert_name, config=config_bert)
        self.emb_dropout_text = nn.Dropout(config.emb_dropout_text)
        self.fc_text = nn.Linear(config.dim_text, config.dim_fusion)
        self.pos_embedding_text = nn.Parameter(torch.randn(1, config.sequence_size_text, config.dim_fusion))

        net = models.densenet121(pretrained=True)
        self.densenet_img = net.features
        self.Avgpool = nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size))
        self.fc_img = nn.Linear(config.dim_img, config.dim_fusion)
        self.emb_dropout_img = nn.Dropout(config.emb_dropout_img)
        self.pos_embedding_img = nn.Parameter(torch.randn(1, config.img_lenth, config.dim_fusion))

        group_matrix = create_group_matrix([], config.tabular_size)
        tabnet = TabNet(config.tabular_size, config.num_classes,cat_emb_dim=[], 
                             group_attention_matrix=group_matrix.to(config.device))
        self.embedding_tabular = tabnet.embedder
        self.tabNetencoder_tabular = tabnet.encoder
        # self.fc_tabular = nn.Linear(config.tabnet_n_d,config.dim_fusion)
        self.emb_dropout_tabular = nn.Dropout(config.emb_dropout_tabular)
        self.pos_embedding_tabular = nn.Parameter(torch.randn(1, config.tabnet_n_steps, config.dim_fusion))


        self.transformer_fusion = Transformer_fusion_missing(config.fusion_lenth, config.dim_fusion, config.depth, config.heads, config.dim_feedforward, config.dropout, config.activation, config.layer_norm_eps, config.batch_first, config.norm_first, config.pool_fusion)
        self.pool = config.pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_fusion, config.num_classes)
        )

        # for p in self.bert.parameters():
        #     p.requires_grad = freeze_bert
    def forward(self, input_ids, attention_mask, img, tabular):
        if input_ids is not None:
            text = self.bert_text(input_ids, attention_mask)
            text = text[0]
            text = self.fc_text(text)
            text = self.emb_dropout_text(text)
            b, c_text, _ = text.shape

            text += self.pos_embedding_text[:, :c_text] 
        else:
            text = None
        if img is not None:
            img = self.densenet_img(img)
            img = self.Avgpool(img)
            img = rearrange(img, 'a b c d -> a (c d) b')
            b, c_img, _ = img.shape
            img = self.fc_img(img)
            img = self.emb_dropout_img(img)
            img += self.pos_embedding_img[:, :c_img]
        if tabular is not None:
            tabular = self.embedding_tabular(tabular)
            tabular, M_loss = self.tabNetencoder_tabular(tabular)

            tabular = torch.stack(tabular, dim=0)

            tabular = rearrange(tabular, 'h b n -> b h n')
            b, c_tabular, _ = tabular.shape
            tabular = self.emb_dropout_tabular(tabular)
            tabular += self.pos_embedding_tabular[:, :c_tabular]


        fusion_token = self.transformer_fusion(img, text, tabular)
        if self.pool == 'mean':
            fusion_token = fusion_token.mean(dim=1)
        elif self.pool == 'cls':
            fusion_token = fusion_token[:, 0]
        elif self.pool == 'last':
            fusion_token = fusion_token[:, -1]

        fusion_token = self.to_latent(fusion_token)
        return self.mlp_head(fusion_token), fusion_token
    
class TEXT_IMG_Tabular_SA2(nn.Module): #三个不同的初始token,与缺失模态初始token池化
    def __init__(self, config):
        super().__init__()

        self.config = config
        config_bert = AutoConfig.from_pretrained(config.bert_name)
        config_bert.update({'attention_probs_dropout_prob': config.dropout, 'hidden_dropout_prob': config.dropout})
        self.bert_text = AutoModel.from_pretrained(config.bert_name, config=config_bert)
        self.emb_dropout_text = nn.Dropout(config.emb_dropout_text)
        self.fc_text = nn.Linear(config.dim_text, config.dim_fusion)
        self.pos_embedding_text = nn.Parameter(torch.randn(1, config.sequence_size_text, config.dim_fusion))

        net = models.densenet121(pretrained=True)
        self.densenet_img = net.features
        self.Avgpool = nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size))
        self.fc_img = nn.Linear(config.dim_img, config.dim_fusion)
        self.emb_dropout_img = nn.Dropout(config.emb_dropout_img)
        self.pos_embedding_img = nn.Parameter(torch.randn(1, config.img_lenth, config.dim_fusion))

        tabnet = torch.load(
            '../saved_model/tabnet_%s_%s_seed%s_%s.pth'%(config.tabnet_n_steps, config.tabnet_n_d, config.random_seed, config.data_from), map_location=torch.device('cpu'))
        self.embedding_tabular = tabnet.embedder
        self.tabNetencoder_tabular = tabnet.tabnet.encoder
        # self.fc_tabular = nn.Linear(config.tabnet_n_d,config.dim_fusion)
        self.emb_dropout_tabular = nn.Dropout(config.emb_dropout_tabular)
        self.pos_embedding_tabular = nn.Parameter(torch.randn(1, config.tabnet_n_steps, config.dim_fusion))


        self.transformer_fusion = Transformer_fusion_missing(config.fusion_lenth, config.dim_fusion, config.depth, config.heads, config.dim_feedforward, config.dropout, config.activation, config.layer_norm_eps, config.batch_first, config.norm_first, config.pool_fusion)
        self.pool = config.pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_fusion, config.num_classes)
        )
        # 是否冻结bert，不让其参数更新
        # for p in self.bert.parameters():
        #     p.requires_grad = freeze_bert
    def forward(self, input_ids, attention_mask, img, tabular):
        if input_ids is not None:
            text = self.bert_text(input_ids, attention_mask)
            text = text[0]
            text = self.fc_text(text)
            text = self.emb_dropout_text(text)
            b, c_text, _ = text.shape
            # x = torch.unsqueeze(x, 2)   #输入特征是1维的，所以要增添这一维度,否则不用
            text += self.pos_embedding_text[:, :c_text]  # 加位置编码
        else:
            text = None
        if img is not None:
            img = self.densenet_img(img)
            img = self.Avgpool(img)
            img = rearrange(img, 'a b c d -> a (c d) b')
            b, c_img, _ = img.shape
            img = self.fc_img(img)
            img = self.emb_dropout_img(img)
            img += self.pos_embedding_img[:, :c_img]
        if tabular is not None:
            tabular = self.embedding_tabular(tabular)
            tabular, M_loss = self.tabNetencoder_tabular(tabular)
            # 后面接个sa试试
            tabular = torch.stack(tabular, dim=0)
            # tabular = self.fc_tabular(tabular)
            tabular = rearrange(tabular, 'h b n -> b h n')
            b, c_tabular, _ = tabular.shape
            tabular = self.emb_dropout_tabular(tabular)
            tabular += self.pos_embedding_tabular[:, :c_tabular]


        fusion_token = self.transformer_fusion(img, text, tabular)
        if self.pool == 'mean':
            fusion_token = fusion_token.mean(dim=1)
        elif self.pool == 'cls':
            fusion_token = fusion_token[:, 0]
        elif self.pool == 'last':
            fusion_token = fusion_token[:, -1]

        fusion_token = self.to_latent(fusion_token)
        return self.mlp_head(fusion_token), fusion_token
