import os
import sys
import time
import numpy as np
import json
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_scheduler, setup_logger
from get_options import parse_option
from module.horizonnet.layout_estimation import HorizonNet
import torch.nn.functional as F

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'igibson':
        from igibson.igbson_detection_dataloader import IGbsonDetectionDataset
        from igibson.model_util_igbson import IGbsonDatasetConfig

        DATASET_CONFIG = IGbsonDatasetConfig()
        TRAIN_DATASET = IGbsonDetectionDataset('train', num_points=args.num_point,
                                                 augment=False,
                                                 use_color=True if args.use_color else False,
                                                 use_height=True if args.use_height else False,
                                                 use_v1=(not args.use_sunrgbd_v2),
                                                 ROOT_DIR = args.igibson_root_dir,
                                                 latent_code_dim=args.emb_dim)

        TEST_DATASET = IGbsonDetectionDataset('val', num_points=args.num_point,
                                                    augment=False,
                                                    use_color=True if args.use_color else False,
                                                    use_height=True if args.use_height else False,
                                                    use_v1=(not args.use_sunrgbd_v2),
                                                    ROOT_DIR=args.igibson_root_dir,
                                                    latent_code_dim=args.emb_dim)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    print(f"train_len: {len(TRAIN_DATASET)}, test_len: {len(TEST_DATASET)}")

    print("training data shuffle: ",True if torch.cuda.is_available() else False)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                               batch_size=args.batch_size if torch.cuda.is_available() else 1,
                                               shuffle=True if torch.cuda.is_available() else False,
                                               num_workers=args.num_workers,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               drop_last=False)

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)
    print(f"train_loader_len: {len(train_loader)}, test_loader_len: {len(test_loader)}")

    return train_loader, test_loader, DATASET_CONFIG

def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False, save_best = False):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_best:
        state['save_path'] = os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'best_valid_epoch_{epoch}.pth')))

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        state['save_path'] = 'current.pth'
        torch.save(state, os.path.join(args.log_dir, 'current.pth'))
        pass

class BaseLoss(object):
    '''base loss class'''
    def __init__(self, config=None):
        '''initialize loss module'''
        self.config = config

    def __call__(self, est_data, gt_data):
        return {}

class HorizonLoss(BaseLoss):
    def __call__(self, est_data, gt_data, prefixe = ''):
        losses = {}
        losses[prefixe+'bon_loss'] = F.l1_loss(est_data[prefixe +'bon'], gt_data['bon'])
        losses[prefixe+'cor_loss'] = F.binary_cross_entropy_with_logits(est_data[prefixe+'cor'], gt_data['cor'])
        losses[prefixe+'layout_loss'] = losses[prefixe+'bon_loss'] + losses[prefixe+'cor_loss']

        return losses

def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, config):
    model.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        end_points = model(batch_data_label['image'])

        # Compute loss and gradients, update parameters.
        loss = criterion(end_points, batch_data_label, prefixe='initial_')

        optimizer.zero_grad()
        loss['initial_'+'layout_loss'].backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate statistics and print out

        if (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {loss[key].item()} \t' for key in loss]))
            logger.info('grad_norm: {}'.format(grad_total_norm.item()))

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in loss.items():
            k = 'train/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

def evaluate_one_epoch(epoch, test_loader, model, criterion, config):

    model.eval()  # set model to training mode
    eval_loss = 0
    for batch_idx, batch_data_label in enumerate(test_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        with torch.no_grad():
            end_points = model(batch_data_label['image'])

        # Compute loss and gradients, update parameters.
        loss = criterion(end_points, batch_data_label, prefixe='initial_')
        eval_loss = eval_loss + loss['initial_'+'layout_loss'].item()

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in loss.items():
            k = 'val/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

        # Accumulate statistics and print out
        if (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Eval: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {loss[key].item()} \t' for key in loss]))


    return eval_loss


if __name__ == '__main__':
    args = parse_option()


    if(torch.cuda.is_available()):
        # model = HorizonNet(cfg=args, pretrain_ckpt='/mnt/workspace/code/DeepPanoContext/pretratin_weight/HorizonNet/resnet50_rnn__st3d.pth')
        model = HorizonNet(cfg=args)
        model = model.cuda()
    else:
        model = HorizonNet(cfg=args, pretrain_ckpt='/Users/yuandong/Documents/Git_project_DAMO/DeepPanoContext_PAI/pretratin_weight/HorizonNet/resnet50_rnn__st3d.pth')

    train_loader, test_loader, DATASET_CONFIG = get_loader(args)

    LOG_DIR = os.path.join(args.log_dir, 'layout_estimation_nopretrain',
                           f'{args.dataset}_{int(time.time())}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(args.log_dir, 'layout_estimation_nopretrain',
                               f'{args.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name="layout_estimation_nopretrain")
    path = os.path.join(args.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(args)))

    tb_writer = SummaryWriter(log_dir=LOG_DIR)

    criterion = HorizonLoss()

    if(args.optimizer == 'adam'):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif(args.optimizer == 'adamW'):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        print("unkown optimizer!")


    scheduler = get_scheduler(optimizer, len(train_loader), args)

    val_best = 1e5

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        tic = time.time()

        train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion, optimizer, scheduler, args)

        if(len(optimizer.param_groups)>1):
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}, lr_decoder {:.5f}'.format(epoch, (time.time() - tic),
                                                                   optimizer.param_groups[0]['lr'],
                                                                   optimizer.param_groups[1]['lr']))
        else:
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}'.format(epoch, (time.time() - tic),optimizer.param_groups[0]['lr']))

        if(epoch%args.val_freq==0):
            val_best_update = evaluate_one_epoch(epoch,test_loader, model, criterion, args)
            if(val_best_update<val_best):
                val_best = val_best_update
                save_checkpoint(args, epoch, model, optimizer, scheduler, save_best=True)

        if True:
            # save model
            save_checkpoint(args, epoch, model, optimizer, scheduler)
    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))

