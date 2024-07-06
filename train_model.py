# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from nets.LViT import LViT
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, get_cosine_schedule_with_warmup, \
            WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile

#import sam
from local_segment_anything import sam_model_registry
from augmentation import get_augmentation_gray
import shutil
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{}.pth.tar'.format(model, "last")
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = get_augmentation_gray([config.img_size, config.img_size], train_flag=True)
    val_tf = get_augmentation_gray([config.img_size, config.img_size], train_flag=False)
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)
    elif config.task_name == 'Covid19' or config.task_name == 'MosMed':

        text_train = config.text_train
        train_dataset = ImageToImage2D(config.train_dataset, text_train, config.task_name, train_tf,
                                       image_size=config.img_size, mean_text_flag=config.mean_text_flag)
        text_val = config.text_val
        val_dataset = ImageToImage2D(config.val_dataset, text_val, config.task_name, val_tf, 
                                     image_size=config.img_size, mean_text_flag=config.mean_text_flag)
      

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=2,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
        pretrained_UNet_model_path = "MoNuSeg/LViT/Test_session_05.23_10h55/models/best_model-LViT.pth.tar"
        pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda')
        pretrained_UNet = pretrained_UNet['state_dict']
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
        print(state_dict.keys())
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        logger.info('Load successful!')
    elif model_type == 'sam':
        config_sam = config.get_sam_config()
        model = sam_model_registry[config_sam.model_type](checkpoint=config_sam.checkpoint, adapter_flag=config_sam.adapter, 
                                                        interaction_indexes=config_sam.interaction_indexes,
                                                        adapter_num_heads=config_sam.num_heads,
                                                        downsample_rate=config_sam.downsample_rate,
                                                        cff_ratio=config_sam.cff_ratio)
        
        for name, para in model.named_parameters():
            if 'mask_decoder' in name:
                para.requires_grad_(True)
                
            elif 'prompt_encoder' in name:
                para.requires_grad_(False)
            elif "image_encoder" in name and ("interaction" not in name and 'segmentic_token' not in name and 'prompt_generator' not in name and 'Adapter' not in name and 'text_convert' not in name):
                if config_sam.finetune_all:
                    para.requires_grad_(True)
                else:
                    para.requires_grad_(False)
            else:
                para.requires_grad_(True)
    else:
        raise TypeError('Please enter a valid name for the model type')
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('model_grad_params: %.2f M'%(model_grad_params/1e6))
    logger.info('model_total_params: %.2f M'%(model_total_params/1e6))

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    criterion = WeightedDiceBCE(dice_weight=1, BCE_weight=1)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.001)  # Choose optimize
    if config.cosineLR is True:
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=lr/100)
        num_training_step = len(train_loader) * config.epochs
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_training_steps= num_training_step, 
                                                    num_warmup_steps=config.warm_iter, )
    else:
        lr_scheduler = None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # load pretrained model
    if os.path.exists(config.pretrained_model):
        logger.info('\n========= load model from {} ========='.format(config.pretrained_model))
        checkpoint = torch.load(config.pretrained_model)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        previous_epoch = checkpoint['epoch']
    else:
        previous_epoch = -1

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(previous_epoch+1, config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_loss, train_dice = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger, 'train')  # sup

        logger.info('Train epoch %d: loss(%.4f) Dice(%.4f)'%(epoch, train_loss, train_dice))
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, None, model_type, logger, 'val')
        logger.info('Valid epoch %d: loss(%.4f) Dice(%.4f)'%(epoch, val_loss, val_dice))
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            # if epoch + 1 > 5:
            logger.info(
                '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, test_dice))
            max_dice = val_dice
            best_epoch = epoch + 1
            save_checkpoint({'epoch': epoch,
                                'best_model': True,
                                'model': model_type,
                                'state_dict': model.state_dict(),
                                'val_loss': val_loss,
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler':lr_scheduler.state_dict()}, config.model_path)
        
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    import shutil
    shutil.copy('Config.py', '%s/Config_ori.py'%(config.save_path))
    # copy xlsx file

    logger = logger_config(log_path=config.logger_path)
    model = main_loop(model_type=config.model_name, tensorboard=True)
