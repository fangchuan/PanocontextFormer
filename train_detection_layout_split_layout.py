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
from models import get_loss
import torch.nn.functional as F
from models import APCalculator, parse_predictions, parse_groundtruths
from tqdm import tqdm

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

def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion_obj, criterion_layout, optimizer, scheduler, config):
    model.train()  # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        end_points = model(batch_data_label)

        for key in batch_data_label:
            if (key == 'scan_name'):
                continue
            end_points[key] = batch_data_label[key]

        # Compute loss and gradients, update parameters.
        # layout-loss
        loss, end_points = criterion_obj(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     box_loss_coef=config.box_loss_coef,
                                     sem_cls_loss_coef=config.sem_cls_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     size_loss_type=config.size_loss_type,
                                     size_delta=config.size_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     heading_delta=config.heading_delta,
                                     size_cls_agnostic=config.size_cls_agnostic,
                                     emb_code_delta = config.emb_code_delta)

        if config.layout_flag:
            initial_layout_loss = criterion_layout(end_points, batch_data_label, prefixe='initial_')
            refine_layout_loss = criterion_layout(end_points, batch_data_label, prefixe='refine_')

        end_points['total_loss'] = (initial_layout_loss['initial_layout_loss']+refine_layout_loss['refine_layout_loss'])/2 + end_points['object_loss']

        optimizer.zero_grad()
        end_points['total_loss'].backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate statistics and print out

        if (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {initial_layout_loss[key].item()} \t' for key in initial_layout_loss]))
            logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {refine_layout_loss[key].item()} \t' for key in refine_layout_loss]))
            logger.info('grad_norm: {}'.format(grad_total_norm.item()))

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in end_points.items():
            if('loss' in k):
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), cur_iter)

        for k, v in initial_layout_loss.items():
            k = 'train/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

        for k, v in refine_layout_loss.items():
            k = 'train/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

def evaluate_one_epoch(epoch, test_loader, model, criterion_obj, criterion_layout, config):

    model.eval()  # set model to training mode
    eval_loss = 0
    AP_IOU_THRESHOLDS = [0.15]

    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,
                   'dataset_config': DATASET_CONFIG}

    if config.num_decoder_layers > 0:
        prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)] + ['layout_refine_']
    else:
        prefixes = ['proposal_']  # only proposal
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        with torch.no_grad():
            end_points = model(batch_data_label)

        for key in batch_data_label:
            if (key == 'scan_name'):
                continue
            end_points[key] = batch_data_label[key]

        # Compute loss and gradients, update parameters.
        loss, end_points = criterion_obj(end_points, DATASET_CONFIG,
                                     num_decoder_layers=config.num_decoder_layers,
                                     query_points_generator_loss_coef=config.query_points_generator_loss_coef,
                                     obj_loss_coef=config.obj_loss_coef,
                                     box_loss_coef=config.box_loss_coef,
                                     sem_cls_loss_coef=config.sem_cls_loss_coef,
                                     query_points_obj_topk=config.query_points_obj_topk,
                                     center_loss_type=config.center_loss_type,
                                     center_delta=config.center_delta,
                                     size_loss_type=config.size_loss_type,
                                     size_delta=config.size_delta,
                                     heading_loss_type=config.heading_loss_type,
                                     heading_delta=config.heading_delta,
                                     size_cls_agnostic=config.size_cls_agnostic,
                                     emb_code_delta = config.emb_code_delta)

        if config.layout_flag:
            initial_layout_loss = criterion_layout(end_points, batch_data_label, prefixe='initial_')
            refine_layout_loss = criterion_layout(end_points, batch_data_label, prefixe='refine_')

        end_points['total_loss'] = (initial_layout_loss['initial_'+'layout_loss']+refine_layout_loss['refine_'+'layout_loss'])/2 + end_points['object_loss']

        eval_loss = eval_loss + end_points['total_loss'].item()

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in end_points.items():
            if('loss' in k):
                k = 'val/%s' % k
                tb_writer.add_scalar(k, v.item(), cur_iter)

        for k, v in initial_layout_loss.items():
            k = 'val/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

        for k, v in refine_layout_loss.items():
            k = 'val/%s' % k
            tb_writer.add_scalar(k, v.item(), cur_iter)

        for prefix in prefixes:
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix,
                                                   size_cls_agnostic=config.size_cls_agnostic)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT,
                                                  size_cls_agnostic=config.size_cls_agnostic)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

    mAP = 0.0
    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            logger.info(f'=====================>{prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<=====================')
            for key in metrics_dict:
                logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

    for mAP in mAPs:
        logger.info(
            f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

        for key in sorted(mAP[1].keys()):
            k = 'mAP/%s' % key
            tb_writer.add_scalar(k, mAP[1][key].item(), epoch)

    return mAP, mAPs

def get_model(config, DATASET_CONFIG):
    from models.detector_layout import GroupFreeDetectorHorizonNet
    num_input_channel = int(args.use_color) * 3

    model = GroupFreeDetectorHorizonNet(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=config.width,
                              bn_momentum=config.bn_momentum,
                              sync_bn=True if config.syncbn else False,
                              num_proposal=config.num_target,
                              sampling=config.sampling,
                              dropout=config.transformer_dropout,
                              activation=config.transformer_activation,
                              nhead=config.nhead,
                              num_decoder_layers=config.num_decoder_layers,
                              dim_feedforward=config.dim_feedforward,
                              self_position_embedding=config.self_position_embedding,
                              cross_position_embedding=config.cross_position_embedding,
                              size_cls_agnostic=True if config.size_cls_agnostic else False,
                              emb_codes_dim=config.emb_dim,
                              image_feature_fusion = config.image_feature_fusion,
                              layout_flag=config.layout_flag)

    return model


if __name__ == '__main__':
    args = parse_option()

    train_loader, test_loader, DATASET_CONFIG = get_loader(args)

    model = get_model(config = args, DATASET_CONFIG = DATASET_CONFIG)

    if(torch.cuda.is_available()):
        model = model.cuda()

    LOG_DIR = os.path.join(args.log_dir, 'layout_detection', f'{args.dataset}_{int(time.time())}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(args.log_dir, 'layout_detection', f'{args.dataset}_{int(time.time())}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name="layout_detection")
    path = os.path.join(args.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(args)))

    tb_writer = SummaryWriter(log_dir=LOG_DIR)

    criterion_obj = get_loss
    criterion_layout = HorizonLoss()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad and "layout_estimation_net" not in n]},
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad and "layout_estimation_net" not in n],
            "lr": args.decoder_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad and "layout_estimation_net" in n],
            "lr": 0.0002,
        },
    ]
    if(args.optimizer == 'adam'):
        optimizer = optim.Adam(param_dicts,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif(args.optimizer == 'adamW'):
        optimizer = optim.AdamW(param_dicts,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        print("unkown optimizer!")


    scheduler = get_scheduler(optimizer, len(train_loader), args)

    val_best = 1e5

    for epoch in range(args.start_epoch, args.max_epoch + 1):

        tic = time.time()

        train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion_obj, criterion_layout, optimizer, scheduler, args)

        if(len(optimizer.param_groups)>1):
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}, lr_decoder {:.5f}, lr_layout {:.5f},'.format(epoch, (time.time() - tic),
                                                                   optimizer.param_groups[0]['lr'],
                                                                   optimizer.param_groups[1]['lr'],
                                                                   optimizer.param_groups[2]['lr']))
        else:
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}'.format(epoch, (time.time() - tic),optimizer.param_groups[0]['lr']))

        if(epoch%args.val_freq==0):
            evaluate_one_epoch(epoch,test_loader, model, criterion_obj, criterion_layout, args)
            save_checkpoint(args, epoch, model, optimizer, scheduler, save_best=True)

    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))

