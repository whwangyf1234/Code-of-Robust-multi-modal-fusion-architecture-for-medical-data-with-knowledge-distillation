import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import re
import allpremodel
import allmodel
import random
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import time
from config import Config
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
import os
def train_tabnet(struct_data, label, config):
    model_path = '../saved_model/tabnet_%s_%s_seed%s_%s.pth'%(config.tabnet_n_steps, config.tabnet_n_d, config.random_seed, config.data_from)
    if os.path.isfile(model_path):
        print("The tabnet file exists!")
        return
    else:
        print("The file does not exist!")
    lenth = label.shape[0]
    label = label.reshape(lenth, 1)
    index = list(np.arange(lenth))
    random.seed(config.random_seed)
    train_index = random.sample(index, round(lenth * config.rat))
    left_index = [x for x in index if x not in train_index]
    lenth_left = len(left_index)
    val_index = random.sample(left_index, round(lenth_left * 0.5))
    test_index = [x for x in left_index if x not in val_index]

    label_train = label[train_index]
    tabular_train = struct_data[train_index]

    label_val = label[val_index]
    tabular_val = struct_data[val_index]

    label_test = label[test_index]
    tabular_test = struct_data[test_index]

    label_train = label_train.flatten()
    label_val = label_val.flatten()

    clf = TabNetClassifier(n_d=config.tabnet_n_d, n_a=config.tabnet_n_a,
                           n_steps=config.tabnet_n_steps)  # TabNetRegressor()
    clf.fit(
        tabular_train, label_train,
        eval_set=[(tabular_val, label_val)]
    )
    preds = clf.predict_proba(tabular_test)[:, 1]
    AUC = metrics.roc_auc_score(label_test, preds)
    a = clf.network
    torch.save(a, model_path)
def tokenizer1(notes_dataframe):  # 去除标点符号，分词
    notes_dataframe = notes_dataframe.apply(lambda x: re.sub(r"[\s;'\"\",_().!?\\/\[\]<>=-]+", ' ', x)).copy()
    notes_dataframe = notes_dataframe.str.replace(r'\s{2,}', ' ', regex=True).str.strip()
    return list(notes_dataframe)


class data_Dataset(Dataset):

    def __init__(self, token, text, struct_data, label, config, data_name):
        image = torch.load(config.image_path)
        # --------------------------
        # ---- 划分训练集 ----
        # --------------------------
        lenth = label.shape[0]
        # label = label.reshape(lenth, 1)
        index = list(np.arange(lenth))
        random.seed(config.random_seed)
        train_index = random.sample(index, round(lenth * config.rat))
        left_index = [x for x in index if x not in train_index]
        lenth_left = len(left_index)
        val_index = random.sample(left_index, round(lenth_left * 0.5))
        test_index = [x for x in left_index if x not in val_index]

        if data_name == 'train':
            self.label = label[train_index]
            self.image = image[train_index]
            self.tabular = struct_data[train_index]
            text = text[train_index]
        elif data_name == 'val':
            self.label = label[val_index]
            self.image = image[val_index]
            self.tabular = struct_data[val_index]
            text = text[val_index]
        elif data_name == 'test':
            self.label = label[test_index]
            self.image = image[test_index]
            self.tabular = struct_data[test_index]
            text = text[test_index]
        if config.selftoken:
            text = tokenizer1(text)
        else:
            text = list(text)
        text = token(text, truncation=True, padding=True, max_length=config.sequence_size_text, add_special_tokens=True)
        self.textids = torch.tensor(text['input_ids'])
        self.atten_mask = torch.tensor(text['attention_mask'])

    def __getitem__(self, index):
        label = self.label[index]
        textids = self.textids[index]
        atten_mask = self.atten_mask[index]
        img = self.image[index]
        tabular = self.tabular[index]
        return label, textids, atten_mask, img, tabular

    def __len__(self):
        return len(self.label)


def train_TEXT_net1(
        text_image_list,  # 图像path，text文字，label
        struct_data,
        label,
        token,
        config
):
    # 构造vit模型
    dataset_train = data_Dataset(token,
                                 text_image_list,  # 图像path，text文字，label
                                 struct_data,
                                 label,
                                 config,
                                 data_name="train"  # 定义训练集或测试集 train or test
                                 )  # 训练集比例
    dataset_val = data_Dataset(token,
                               text_image_list,  # 图像path，text文字，label
                               struct_data,
                               label,
                               config,
                               data_name="val",  # 定义训练集或测试集 train or test
                               )  # 训练集比例
    loader_train = DataLoader(  # 批训练数据加载器
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,  # 每次训练打乱数据， 默认为False
        num_workers=32,  # 使用多进行程读取数据， 默认0，为不使用多进程
        pin_memory=True,
        drop_last=True
    )
    loader_val = DataLoader(  # 批训练数据加载器
        dataset=dataset_val,
        batch_size=32,  # len(dataset_val)
        shuffle=False,  # 每次训练打乱数据， 默认为False
        num_workers=32,  # 使用多进行程读取数据， 默认0，为不使用多进程
        pin_memory=True,
        drop_last=False
    )
    device = config.device
    model = allmodel.TEXT_IMG_Tabular_SA1(config).to(device)
    # model = torch.compile(model)
    # print(model)

    # Loss and optimizer
    criterion1 = nn.CrossEntropyLoss()
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': config.bert_rate,
         'weight_decay': config.bert_decay},
        {'params': [p for n, p in model.named_parameters() if 'densenet' in n], 'lr': config.img_rate,
         'weight_decay': config.img_decay},
        {'params': [p for n, p in model.named_parameters() if 'mean_weights' in n], 'lr': 0.01,
         'weight_decay': config.img_decay},
        {'params': [p for n, p in model.named_parameters() if ('bert' not in n and 'densenet' not in n and 'mean_weights' not in n)],'lr': config.learning_rate,
         'weight_decay': config.weight_decay}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    # optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(
    #   optimizer,
    #   num_warmup_steps=0,
    #   num_training_steps=20
    # )
    # 训练模型
    total_step = math.ceil(len(dataset_train) / config.batch_size)
    loss_train = []
    loss_val = []
    loss_min = 100
    sum_loss = 0
    stay_times = 0  # 当前stay_time
    stay_time = config.stay_time  # 终止stay_time
    step_print = config.step_print  # 终止step
    rastay_times = 0  # 当前rastay_time
    rastay_time = config.rastay_time  # 终止rastay_time
    start_time = time.time()
    for epoch in range(config.num_epochs):
        for i, (labels, textids, atten_mask, img, tabular) in enumerate(loader_train):
            # 前向传播
            # print(i)
            model.train()
            optimizer.zero_grad()
            labels = labels.to(device)
            img = img.to(device)
            textids = textids.to(device)
            atten_mask = atten_mask.to(device)
            tabular = tabular.to(torch.float32).to(device)

            outputs, _ = model(textids, atten_mask, img, tabular)
            loss = criterion1(outputs, labels)
            loss.backward()

            # outputs_text, _ = model(textids, atten_mask, None, None)
            # outputs_image, _ = model(None, atten_mask, img, None)
            # outputs_tabular, _ = model(None, atten_mask, None, tabular)
            #
            # score_text = sum([softmax(outputs_text)[i][labels[i]] for i in range(config.batch_size)])
            # score_image = sum([softmax(outputs_image)[i][labels[i]] for i in range(config.batch_size)])
            # score_tabular = sum([softmax(outputs_tabular)[i][labels[i]] for i in range(config.batch_size)])

            # ratio_text = score_text/(score_image+score_tabular)
            # ratio_image = score_image/(score_text+score_tabular)
            # ratio_tabular = score_tabular/(score_image+score_text)

            a = model.transformer_fusion.mean_weights
            b = (1-(a / torch.sum(a))).tolist()
            coeff_text = b[0]  # coeff_text = 1 - sigmoid(config.alpha * log(ratio_text))
            coeff_image = b[1]
            coeff_tabular = b[2]

            if config.modulation_starts <= epoch <= config.modulation_ends: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name)

                    if 'text' in layer and parms.grad is not None :
                        if config.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_text + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)    #加高斯噪声
                        elif config.modulation == 'OGM':
                            parms.grad *= coeff_text

                    if 'image' in layer:    #维度为4是为什么？
                        if config.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_image + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif config.modulation == 'OGM':
                            parms.grad *= coeff_image

                    if 'tabular' in layer:    #维度为4是为什么？
                        if config.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_tabular + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif config.modulation == 'OGM':
                            parms.grad *= coeff_tabular
            else:
                pass

            sum_loss += loss.item()
            # 后向优化

            optimizer.step()

            # 每一百步打印一次
            step_prints = step_print
            if (i + 1) % step_prints == 0:
                # torch.cuda.empty_cache()
                model.eval()
                loss_train.append(sum_loss / step_prints)

                with torch.no_grad():
                    loss_v = []
                    for j, (labels, textids, atten_mask, img, tabular) in enumerate(loader_val):
                        labels = labels.to(device)
                        img = img.to(device)
                        textids = textids.to(device)
                        atten_mask = atten_mask.to(device)
                        tabular = tabular.to(torch.float32).to(device)
                        outputs, _ = model(textids, atten_mask, img, tabular)
                        loss_v_ = criterion1(outputs, labels)
                        loss_v.append(loss_v_.item())
                    loss_v = np.mean(loss_v)
                    if loss_v < loss_min:
                        loss_min = loss_v.item()
                        torch.save(model, '../saved_model/%s.pth' % config.model_save_path)
                        stay_times = 0
                        rastay_times = 0
                    elif loss_v > loss_min:
                        stay_times += 1
                        if stay_times == stay_time:
                            stay_times = 0
                            rastay_times += 1
                            scheduler.step()

                loss_val.append(loss_v)
                plt.close('all')
                plt.figure(config.model_save_path)
                plt.title(config.model_save_path)
                plt.plot(range(len(loss_val)), loss_val, label='loss_val')
                plt.plot(range(len(loss_val)), loss_train, label='loss_train', color="red", linewidth=1.0,
                         linestyle='--')
                plt.legend()

                end_time = time.time()
                time1 = end_time - start_time
                start_time = time.time()

                print('valloss_min:%4f' % loss_min)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},学习率：{},{},{},weight_decay:{},{},{}, time:{}'
                      .format(epoch + 1, config.num_epochs, i + 1, total_step, sum_loss / step_print,
                              optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'],
                              optimizer.param_groups[0]['weight_decay'], optimizer.param_groups[1]['weight_decay'], optimizer.param_groups[2]['weight_decay'],
                              time1))
                sum_loss = 0

        if rastay_times >= rastay_time:
            break

    print("训练完毕")
    # 保存训练完的模型

def train_TEXT_net2(
        text_image_list,  # 图像path，text文字，label
        struct_data,
        label,
        token,
        config
):
    # 构造vit模型
    dataset_train = data_Dataset(token,
                                 text_image_list,  # 图像path，text文字，label
                                 struct_data,
                                 label,
                                 config,
                                 data_name="train"  # 定义训练集或测试集 train or test
                                 )  # 训练集比例
    dataset_val = data_Dataset(token,
                               text_image_list,  # 图像path，text文字，label
                               struct_data,
                               label,
                               config,
                               data_name="val",  # 定义训练集或测试集 train or test
                               )  # 训练集比例
    loader_train = DataLoader(  # 批训练数据加载器
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,  # 每次训练打乱数据， 默认为False
        num_workers=32,  # 使用多进行程读取数据， 默认0，为不使用多进程
        pin_memory=True,
        drop_last=True
    )
    loader_val = DataLoader(  # 批训练数据加载器
        dataset=dataset_val,
        batch_size=32,  # len(dataset_val)
        shuffle=False,  # 每次训练打乱数据， 默认为False
        num_workers=32,  # 使用多进行程读取数据， 默认0，为不使用多进程
        pin_memory=True,
        drop_last=False
    )
    device = config.device
    # model = allmodel.TEXT_IMG_Tabular_SA7(config).to(device)
    model = torch.load('../saved_model/%s.pth' % config.model_save_path).to(device)
    # model = torch.compile(model)
    # print(model)

    # Loss and optimizer
    criterion1 = nn.CrossEntropyLoss()
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': config.bert_rate,
         'weight_decay': config.bert_decay},
        {'params': [p for n, p in model.named_parameters() if 'densenet' in n], 'lr': config.img_rate,
         'weight_decay': config.img_decay},
        {'params': [p for n, p in model.named_parameters() if 'mean_weights' in n], 'lr': 0.01,
         'weight_decay': config.img_decay},
        {'params': [p for n, p in model.named_parameters() if ('bert' not in n and 'densenet' not in n and 'mean_weights' not in n)],'lr': config.learning_rate,
         'weight_decay': config.weight_decay}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    # optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(
    #   optimizer,
    #   num_warmup_steps=0,
    #   num_training_steps=20
    # )
    # 训练模型
    total_step = math.ceil(len(dataset_train) / config.batch_size)
    loss_train = []
    loss_val = []
    loss_min = 100
    sum_loss = 0
    stay_times = 0  # 当前stay_time
    stay_time = config.stay_time2  # 终止stay_time
    step_print = config.step_print  # 终止step
    rastay_times = 0  # 当前rastay_time
    rastay_time = config.rastay_time2  # 终止rastay_time
    start_time = time.time()
    for epoch in range(config.num_epochs2):
        for i, (labels, textids, atten_mask, img, tabular) in enumerate(loader_train):
            # 前向传播
            # print(i)
            model.train()
            labels = labels.to(device)
            img = img.to(device)
            textids = textids.to(device)
            atten_mask = atten_mask.to(device)
            tabular = tabular.to(torch.float32).to(device)
            outputs, fusion_token = model(textids, atten_mask, img, tabular)

            inputs = [textids, atten_mask, img, tabular]
            modal_indices = [0, 2, 3]
            random.shuffle(modal_indices)
            num_modal_to_remove = random.randint(1, 2)
            sample = np.random.choice(modal_indices, num_modal_to_remove, replace=False, p=config.p_weight)  # 根据权重从序列中无放回地抽取size个元素
            for j in range(num_modal_to_remove):
                inputs[sample[j]] = None

            outputs1, fusion_token1 = model(inputs[0], atten_mask, inputs[2], inputs[3])
            #distillation_loss_mse
            # loss = criterion1(outputs, labels)+distillation_loss_KL(outputs1, outputs.detach(), config.temperature)  #蒸分类损失KL

            loss = criterion1(outputs, labels) + criterion1(outputs1, torch.softmax(outputs.detach(), dim=-1))  # 蒸分类损失 交叉熵
            # loss = criterion1(outputs, labels) + criterion1(outputs1, labels)  # 蒸分类损失 交叉熵

            # loss = criterion1(outputs, labels) + 10000*distillation_loss_KL(fusion_token1, fusion_token, config.temperature)  #加权蒸8，512token
            # loss = criterion1(outputs, labels) + distillation_loss_KL(fusion_token1, fusion_token, config.temperature)  #蒸 8，10，512token
            sum_loss += loss.item()
            # 后向优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每一百步打印一次
            step_prints = step_print
            if (i + 1) % step_prints == 0:
                # torch.cuda.empty_cache()
                model.eval()
                loss_train.append(sum_loss / step_prints)

                with torch.no_grad():
                    loss_v = []
                    for j, (labels, textids, atten_mask, img, tabular) in enumerate(loader_val):
                        labels = labels.to(device)
                        img = img.to(device)
                        textids = textids.to(device)
                        atten_mask = atten_mask.to(device)
                        tabular = tabular.to(torch.float32).to(device)
                        outputs, _ = model(textids, atten_mask, img, tabular)
                        loss_v_ = criterion1(outputs, labels)
                        loss_v.append(loss_v_.item())
                    loss_v = np.mean(loss_v)
                    if loss_v < loss_min:
                        loss_min = loss_v.item()
                        torch.save(model, '../saved_model/%s.pth' % config.model_save_path)
                        stay_times = 0
                        rastay_times = 0
                    elif loss_v > loss_min:
                        stay_times += 1
                        if stay_times == stay_time:
                            stay_times = 0
                            rastay_times += 1
                            scheduler.step()

                loss_val.append(loss_v)
                plt.close('all')
                plt.figure(config.model_save_path)
                plt.title(config.model_save_path)
                plt.plot(range(len(loss_val)), loss_val, label='loss_val')
                plt.plot(range(len(loss_val)), loss_train, label='loss_train', color="red", linewidth=1.0,
                         linestyle='--')
                plt.legend()

                end_time = time.time()
                time1 = end_time - start_time
                start_time = time.time()

                print('valloss_min:%4f' % loss_min)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},学习率：{},{},{},weight_decay:{},{},{}, time:{}'
                      .format(epoch + 1, config.num_epochs2, i + 1, total_step, sum_loss / step_print,
                              optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'],
                              optimizer.param_groups[0]['weight_decay'], optimizer.param_groups[1]['weight_decay'], optimizer.param_groups[2]['weight_decay'],
                              time1))
                sum_loss = 0

        # torch.save(model, '../saved_model/%s_%s.pth' % (config.model_save_path, epoch+1))
        if rastay_times >= rastay_time:
            break
    # torch.save(model, '../saved_model/%s.pth' % config.model_save_path)

    print("训练完毕")
    # 保存训练完的模型
def test_TEXT_net(text_image_list,  # 图像path，text文字，label
                  struct_data,
                  label,
                  token,
                  config,
                  data_name
                  ):
    dataset = data_Dataset(token,
                           text_image_list,  # 图像path，text文字，label
                           struct_data,
                           label,
                           config,
                           data_name=data_name
                           )  # 训练集比例
    loader = DataLoader(  # 批训练数据加载器
        dataset=dataset,
        batch_size=32,
        shuffle=False,  # 每次训练打乱数据， 默认为False
        num_workers=32,  # 使用多进行程读取数据， 默认0，为不使用多进程
        pin_memory=True,
        drop_last=False
    )
    device = config.device
    model = torch.load('../saved_model/%s.pth' % config.model_save_path).to(device)
    # model = torch.compile(model)
    model.eval()
    # torch.cuda.empty_cache()

    with torch.no_grad():
        # start_time = time.time()
        labels = torch.tensor([])
        predicted = torch.tensor([])
        predictedpro = torch.tensor([])
        predicted1 = torch.tensor([])
        predictedpro1 = torch.tensor([])
        predicted2 = torch.tensor([])
        predictedpro2 = torch.tensor([])
        predicted3 = torch.tensor([])
        predictedpro3 = torch.tensor([])
        predicted4 = torch.tensor([])
        predictedpro4 = torch.tensor([])
        predicted5 = torch.tensor([])
        predictedpro5 = torch.tensor([])
        predicted6 = torch.tensor([])
        predictedpro6 = torch.tensor([])
        for labels_, textids, atten_mask, img, tabular in loader:
            labels_ = labels_.to(device)
            img = img.to(device)
            textids = textids.to(device)
            atten_mask = atten_mask.to(device)
            tabular = tabular.to(torch.float32).to(device)
            outputs, _ = model(textids, atten_mask, img, tabular)
            outputs1, _ = model(None, None, img, tabular)
            outputs2, _ = model(textids, atten_mask, None, tabular)
            outputs3, _ = model(textids, atten_mask, img, None)
            outputs4, _ = model(None, None, None, tabular)
            outputs5, _ = model(None, None, img, None)
            outputs6, _ = model(textids, atten_mask, None, None)

            outputs = F.softmax(outputs, dim=1)  # F.softmax  log_softmax
            predictedpro_ = outputs[:, 1]
            _, predicted_ = torch.max(outputs.data, 1)
            labels = torch.cat((labels, labels_.cpu()))
            predicted = torch.cat((predicted, predicted_.cpu()))
            predictedpro = torch.cat((predictedpro, predictedpro_.cpu()))

            outputs1 = F.softmax(outputs1, dim=1)  # F.softmax  log_softmax
            predictedpro_1 = outputs1[:, 1]
            _, predicted_1 = torch.max(outputs1.data, 1)
            predicted1 = torch.cat((predicted1, predicted_1.cpu()))
            predictedpro1 = torch.cat((predictedpro1, predictedpro_1.cpu()))

            outputs2 = F.softmax(outputs2, dim=1)  # F.softmax  log_softmax
            predictedpro_2 = outputs2[:, 1]
            _, predicted_2 = torch.max(outputs2.data, 1)
            predicted2 = torch.cat((predicted2, predicted_2.cpu()))
            predictedpro2 = torch.cat((predictedpro2, predictedpro_2.cpu()))

            outputs3 = F.softmax(outputs3, dim=1)  # F.softmax  log_softmax
            predictedpro_3 = outputs3[:, 1]
            _, predicted_3 = torch.max(outputs3.data, 1)
            predicted3 = torch.cat((predicted3, predicted_3.cpu()))
            predictedpro3 = torch.cat((predictedpro3, predictedpro_3.cpu()))

            outputs4 = F.softmax(outputs4, dim=1)  # F.softmax  log_softmax
            predictedpro_4 = outputs4[:, 1]
            _, predicted_4 = torch.max(outputs4.data, 1)
            predicted4 = torch.cat((predicted4, predicted_4.cpu()))
            predictedpro4 = torch.cat((predictedpro4, predictedpro_4.cpu()))

            outputs5 = F.softmax(outputs5, dim=1)  # F.softmax  log_softmax
            predictedpro_5 = outputs5[:, 1]
            _, predicted_5 = torch.max(outputs5.data, 1)
            predicted5 = torch.cat((predicted5, predicted_5.cpu()))
            predictedpro5 = torch.cat((predictedpro5, predictedpro_5.cpu()))

            outputs6 = F.softmax(outputs6, dim=1)  # F.softmax  log_softmax
            predictedpro_6 = outputs6[:, 1]
            _, predicted_6 = torch.max(outputs6.data, 1)
            predicted6 = torch.cat((predicted6, predicted_6.cpu()))
            predictedpro6 = torch.cat((predictedpro6, predictedpro_6.cpu()))

        predicted = predicted.numpy()
        labels = labels.numpy()
        predictedpro = predictedpro.numpy()

        predicted1 = predicted1.numpy()
        predictedpro1 = predictedpro1.numpy()
        predicted2 = predicted2.numpy()
        predictedpro2 = predictedpro2.numpy()
        predicted3 = predicted3.numpy()
        predictedpro3 = predictedpro3.numpy()
        predicted4 = predicted4.numpy()
        predictedpro4 = predictedpro4.numpy()
        predicted5 = predicted5.numpy()
        predictedpro5 = predictedpro5.numpy()
        predicted6 = predicted6.numpy()
        predictedpro6 = predictedpro6.numpy()

        ACC = metrics.accuracy_score(labels, predicted)
        AUC = metrics.roc_auc_score(labels, predictedpro)
        PRC = metrics.average_precision_score(labels, predictedpro)
        F1_score = metrics.f1_score(labels, predicted)
        Recall = metrics.recall_score(labels, predicted)
        Precision = metrics.precision_score(y_true=labels, y_pred=predicted)
        meric = [ACC, AUC, PRC, F1_score, Recall, Precision]

        ACC1 = metrics.accuracy_score(labels, predicted1)
        AUC1 = metrics.roc_auc_score(labels, predictedpro1)
        PRC1 = metrics.average_precision_score(labels, predictedpro1)
        F1_score1 = metrics.f1_score(labels, predicted1)
        Recall1 = metrics.recall_score(labels, predicted1)
        Precision1 = metrics.precision_score(y_true=labels, y_pred=predicted1)
        meric1 = [ACC1, AUC1, PRC1, F1_score1, Recall1, Precision1]

        ACC2 = metrics.accuracy_score(labels, predicted2)
        AUC2 = metrics.roc_auc_score(labels, predictedpro2)
        PRC2 = metrics.average_precision_score(labels, predictedpro2)
        F1_score2 = metrics.f1_score(labels, predicted2)
        Recall2 = metrics.recall_score(labels, predicted2)
        Precision2 = metrics.precision_score(y_true=labels, y_pred=predicted2)
        meric2 = [ACC2, AUC2, PRC2, F1_score2, Recall2, Precision2]

        ACC3 = metrics.accuracy_score(labels, predicted3)
        AUC3 = metrics.roc_auc_score(labels, predictedpro3)
        PRC3 = metrics.average_precision_score(labels, predictedpro3)
        F1_score3 = metrics.f1_score(labels, predicted3)
        Recall3 = metrics.recall_score(labels, predicted3)
        Precision3 = metrics.precision_score(y_true=labels, y_pred=predicted3)
        meric3 = [ACC3, AUC3, PRC3, F1_score3, Recall3, Precision3]

        ACC4 = metrics.accuracy_score(labels, predicted4)
        AUC4 = metrics.roc_auc_score(labels, predictedpro4)
        PRC4 = metrics.average_precision_score(labels, predictedpro4)
        F1_score4 = metrics.f1_score(labels, predicted4)
        Recall4 = metrics.recall_score(labels, predicted4)
        Precision4 = metrics.precision_score(y_true=labels, y_pred=predicted4)
        meric4 = [ACC4, AUC4, PRC4, F1_score4, Recall4, Precision4]

        ACC5 = metrics.accuracy_score(labels, predicted5)
        AUC5 = metrics.roc_auc_score(labels, predictedpro5)
        PRC5 = metrics.average_precision_score(labels, predictedpro5)
        F1_score5 = metrics.f1_score(labels, predicted5)
        Recall5 = metrics.recall_score(labels, predicted5)
        Precision5 = metrics.precision_score(y_true=labels, y_pred=predicted5)
        meric5 = [ACC5, AUC5, PRC5, F1_score5, Recall5, Precision5]

        ACC6 = metrics.accuracy_score(labels, predicted6)
        AUC6 = metrics.roc_auc_score(labels, predictedpro6)
        PRC6 = metrics.average_precision_score(labels, predictedpro6)
        F1_score6 = metrics.f1_score(labels, predicted6)
        Recall6 = metrics.recall_score(labels, predicted6)
        Precision6 = metrics.precision_score(y_true=labels, y_pred=predicted6)
        meric6 = [ACC6, AUC6, PRC6, F1_score6, Recall6, Precision6]
        print(
            'ACC: {} %,  AUC: {} %, Pre: {} %, F1: {} %, Recall: {} %'.format(100 * ACC, 100 * AUC, 100 * Precision,
                                                                              100 * F1_score, 100 * Recall))
        print('w/o text')
        print(meric1)
        print('w/o image')
        print(meric2)
        print('w/o tabular')
        print(meric3)
        print('w/o image,text')
        print(meric4)
        print('w/o text, tabular')
        print(meric5)
        print('w/o image, tabular')
        print(meric6)
        # print(time.time()-start_time)
        return meric, meric1, meric2, meric3, meric4, meric5, meric6


if __name__ == '__main__':
    config = Config()
    data_list = pd.read_csv(config.datalist_path)
    text_data = data_list["text"]
    label = np.array(data_list.iloc[:, 4])
    struct_data = data_list.iloc[:, 5:config.tabular_size+5]


    # scaler = StandardScaler()   #正态标准化
    # # scaler = MinMaxScaler(feature_range=(-1, 1)) #最大最小值标准化
    # normalized_data = scaler.fit_transform(struct_data)
    # # 将标准化后的数据重新转换为DataFrame
    # struct_data = pd.DataFrame(normalized_data, columns=struct_data.columns)

    struct_data.fillna(0, axis=1, inplace=True)  #补零
    # struct_data.fillna(struct_data.mean(), inplace=True)  # 补均值

    struct_data = np.array(struct_data)
    # microsoft_Biomed_bert
    token = AutoTokenizer.from_pretrained(config.bert_name)
    metric_train = []
    auc_train = []
    metric_test0 = []
    metric_test1 = []
    metric_test2 = []
    metric_test3 = []
    metric_test4 = []
    metric_test5 = []
    metric_test6 = []
    metric_test_img_text = []
    metric_test_img_struct = []
    metric_test_text_struct = []
    auc_test = []
    Acc_test = []
    prc_test = []
    torch.autograd.set_detect_anomaly(True)
    for config.random_seed in range(1, 11):
        print(config.random_seed)

        # train_tabnet(struct_data, label, config)

        train_TEXT_net1(text_data, struct_data, label, token, config)
        train_TEXT_net2(text_data, struct_data, label, token, config)
        # predicted_, labels_, predictedpro_, metric_ = test_TEXT_net(
        #     text_data,  # 图像path，text文字，label
        #     struct_data,
        #     label,
        #     token,
        #     config,
        #     data_name="train"
        # )
        metric0, metric1, metric2, metric3, metric4, metric5, metric6 = test_TEXT_net(
            text_data,  # 图像path，text文字，label
            struct_data,
            label,
            token,
            config,
            data_name="test"
        )
        # metric_train.append(metric_)
        metric_test0.append(metric0)
        metric_test1.append(metric1)
        metric_test2.append(metric2)
        metric_test3.append(metric3)
        metric_test4.append(metric4)
        metric_test5.append(metric5)
        metric_test6.append(metric6)
    # metric_train = pd.DataFrame(metric_train, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test0 = pd.DataFrame(metric_test0, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test1 = pd.DataFrame(metric_test1, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test2 = pd.DataFrame(metric_test2, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test3 = pd.DataFrame(metric_test3, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test4 = pd.DataFrame(metric_test4, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test5 = pd.DataFrame(metric_test5, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    metric_test6 = pd.DataFrame(metric_test6, columns=['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision'])
    auc_test = pd.DataFrame(auc_test)
    Acc_test = pd.DataFrame(Acc_test)
    prc_test = pd.DataFrame(prc_test)

    modelname = ['all', 'w/o text', 'w/o image', 'w/o tabular', 'w/o image,text', 'w/o text, tabular','w/o image, tabular']

    metric_test = pd.concat([metric_test0, metric_test1, metric_test2, metric_test3, metric_test4, metric_test5, metric_test6], axis=1).T
    metric_test = [metric_test.iloc[i::6] for i in range(6)]
    for df in metric_test:
        df.index = modelname

    sheet_names = ['ACC', 'AUC', 'PRC', 'F1_score', 'Recall', 'Precision']
    with pd.ExcelWriter(r'../result/%s_metric.xlsx' % config.model_save_path, engine='xlsxwriter') as writer:
        # 将每个DataFrame写入不同的工作表
        for i, df in enumerate(metric_test):
            df.to_excel(writer, sheet_name=sheet_names[i])

    metric_test_avg = pd.concat([metric_test0.mean(), metric_test1.mean(), metric_test2.mean(), metric_test3.mean(), metric_test4.mean(), metric_test5.mean(), metric_test6.mean()], axis=1)
    metric_test_avg.columns = modelname
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    print(metric_test_avg)

    plt.show()
