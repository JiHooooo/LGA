# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 50
warm_iter = 1000
img_size = 1024 # sam
# img_size = 224 # norm
print_frequency = 20
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
# task_name = 'MoNuSeg' 
task_name = 'MosMed' # Covid19, MosMed
learning_rate = 1e-4  # MoNuSeg: 1e-3, Covid19: 3e-4
batch_size = 2  # For LViT-T, 2 is better than 4

model_name = 'sam'
# model_name = 'LViT'
# model_name = 'LViT_pretrain'

mean_text_flag = True

train_dataset = ''
val_dataset = ''

text_train = ''
text_val = ''

session_name = 'test' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
# test_session = "Test_session_05.23_14h19"  # dice=79.98, IoU=66.83

##########################################################################
# CTrans configs
##########################################################################
def get_sam_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()

    config.model_type = 'vit_b'
    config.adapter = True
    config.finetune_all = True
    config.adapter_type = 'spa_text'
    config.attn_type = 'global'
    config.num_heads = 1 
    config.downsample_rate = 8 
    config.mlp_vit = 3
    config.cff_ratio = 0.25
    config.interaction_indexes = [0,4,8]
    config.text_cross = True

    config.checkpoint = '/media/iipl/disk3/hu/models/sam_vit_b_01ec64.pth'
    config.seg_head = 'sam'
    config.lora_rank = 0
    return config