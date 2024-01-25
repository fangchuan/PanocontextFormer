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
from module.deep3dlayout.deep3dlayout_model import Deep3DlayoutNet
import torch.nn.functional as F
from module.deep3dlayout.mesh_loss import MeshLoss
from module.deep3dlayout.custom_losses.losses import L_sharp_loss, L_smooth_loss
from pytorch3d.io import save_obj
import torch_mesh_isect.mesh_intersection.loss as collisions_loss
from torch_mesh_isect.mesh_intersection.bvh_search_tree import BVH

def intersection_loss_mesh(pred_mesh, search_tree, pen_distance):

    pred_mesh_verts_pad = pred_mesh.verts_padded()
    pred_mesh_faces_pad = pred_mesh.faces_padded()
    triangles_all = []
    bs = pred_mesh_verts_pad.shape[0]
    for bs_i in range(bs):
        triangles = (pred_mesh_verts_pad[bs_i, ...])[(pred_mesh_faces_pad[bs_i, ...])]
        triangles_all.append(triangles.unsqueeze(0))

    triangles_all_cal = torch.cat(triangles_all, 0)
    collision_idxs = search_tree(triangles_all_cal)
    pen_loss = pen_distance(triangles_all_cal, collision_idxs)
    intersection_loss = torch.sum(pen_loss)/(bs + 1e-6)

    return intersection_loss

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
                                                 augment=True if torch.cuda.is_available() else False,
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
                                               batch_size=args.batch_size if torch.cuda.is_available() else 3,
                                               shuffle=True if torch.cuda.is_available() else False,
                                               num_workers=args.num_workers if torch.cuda.is_available() else 0,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               drop_last=False)

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.num_workers if torch.cuda.is_available() else 0,
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

def load_pretrain_checkpoint(checkpoint_path, model):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("loading Pretrain model... {}".format(checkpoint_path))

    return model


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

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm

def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, mesh_criterion, shape_criterion, smooth_criterion, optimizer, scheduler, config):
    model.train()  # set model to training mode
    search_tree = BVH(max_collisions=8)
    pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.5,
                                                         point2plane=False,
                                                         vectorized=True)

    for batch_idx, batch_data_label in enumerate(train_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        end_points = model(batch_data_label)

        # Compute loss and gradients, update parameters.
        gt_meshes = (batch_data_label['points_gt'],batch_data_label['normals_gt'])

        mesh_loss, losses = mesh_criterion(meshes_pred = end_points['deep3d_meshes'], meshes_gt = gt_meshes)

        # compute sharp-loss
        sharp_loss_0 = shape_criterion(meshes_pred = end_points['deep3d_meshes'][0], meshes_gt = batch_data_label)
        sharp_loss_1 = shape_criterion(meshes_pred = end_points['deep3d_meshes'][1], meshes_gt = batch_data_label)
        losses['sharp_loss_0'] = sharp_loss_0.item()
        losses['sharp_loss_1'] = sharp_loss_1.item()

        smooth_loss_0 = smooth_criterion(end_points['deep3d_meshes'][0])
        smooth_loss_1 = smooth_criterion(end_points['deep3d_meshes'][1])
        losses['smooth_loss_0'] = smooth_loss_0.item()
        losses['smooth_loss_1'] = smooth_loss_1.item()

        intersection_loss_0 = intersection_loss_mesh(end_points['deep3d_meshes'][0], search_tree, pen_distance)
        losses['intersection_loss_0'] = intersection_loss_0

        total_loss = mesh_loss + \
                     (sharp_loss_0 + sharp_loss_1)/len(end_points['deep3d_meshes']) * config.dp3d_sharp_loss_coef + \
                     (smooth_loss_0 + smooth_loss_1)/len(end_points['deep3d_meshes']) * config.dp3d_smooth_loss_coef + \
                     intersection_loss_0

        optimizer.zero_grad()
        total_loss.backward()

        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        losses['total_loss'] = total_loss.item()

        # Accumulate statistics and print out

        if (batch_idx + 1) % 10 == 0:
            logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                [f'{key} {losses[key]} \t' for key in losses]))
            logger.info('grad_norm: {}'.format(grad_total_norm.item()))

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for key in losses:
            k = 'train/%s' % key
            tb_writer.add_scalar(k, losses[key], cur_iter)

def evaluate_one_epoch(epoch, test_loader, model, mesh_criterion, shape_criterion, smooth_criterion, config):

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
            end_points = model(batch_data_label)

        # Compute loss and gradients, update parameters.
        gt_meshes = (batch_data_label['points_gt'],batch_data_label['normals_gt'])

        mesh_loss, losses = mesh_criterion(meshes_pred = end_points['deep3d_meshes'], meshes_gt = gt_meshes)

        # compute sharp-loss
        sharp_loss_0 = shape_criterion(meshes_pred = end_points['deep3d_meshes'][0], meshes_gt = batch_data_label)
        sharp_loss_1 = shape_criterion(meshes_pred = end_points['deep3d_meshes'][1], meshes_gt = batch_data_label)
        losses['sharp_loss_0'] = sharp_loss_0.item()
        losses['sharp_loss_1'] = sharp_loss_1.item()

        smooth_loss_0 = smooth_criterion(end_points['deep3d_meshes'][0])
        smooth_loss_1 = smooth_criterion(end_points['deep3d_meshes'][1])
        losses['smooth_loss_0'] = smooth_loss_0.item()
        losses['smooth_loss_1'] = smooth_loss_1.item()

        os.makedirs(os.path.join(LOG_DIR,'dump_eval'),exist_ok=True)
        output_filepath = os.path.join(LOG_DIR,'dump_eval',str(epoch)+"_"+str(batch_idx))
        if(batch_idx<5):
            save_obj(output_filepath+"_initial.obj", end_points['deep3d_meshes'][0].cpu().detach().verts_packed(), end_points['deep3d_meshes'][0].cpu().detach().faces_packed())
            save_obj(output_filepath+"_refine.obj", end_points['deep3d_meshes'][1].cpu().detach().verts_packed(), end_points['deep3d_meshes'][1].cpu().detach().faces_packed())

        if (batch_idx + 1) % 1 == 0:
            logger.info(f'Eval: [{epoch}][{batch_idx + 1}/{len(test_loader)}]  ' + ''.join(
                [f'{key} {losses[key]} \t' for key in losses]))

        cur_iter = (epoch-1)*len(test_loader)+batch_idx

        for key in losses:
            k = 'Eval/%s' % key
            tb_writer.add_scalar(k, losses[key], cur_iter)


    return eval_loss


if __name__ == '__main__':
    args = parse_option()
    args.max_epoch = 300
    args.val_freq = 20

    if(torch.cuda.is_available()):
        model = Deep3DlayoutNet(backbone='resnet18', decoder_type='rcnn_p2m_mhsa_pos_dual', full_size=True, hidden_dim=128)
        # model = load_pretrain_checkpoint(checkpoint_path='/mnt/workspace/code/deep3dlayout/ckpt/m3d_layout.pth', model = model)
        model = model.cuda()
    else:
        model = Deep3DlayoutNet(backbone='resnet18', decoder_type='rcnn_p2m_mhsa_pos_dual', full_size=True, hidden_dim=128)
        # model = load_pretrain_checkpoint(checkpoint_path='/Users/yuandong/Documents/Git_project_DAMO/Deep3DLayout/ckpt/m3d_layout.pth', model = model)

    train_loader, test_loader, DATASET_CONFIG = get_loader(args)

    LOG_DIR = os.path.join(args.log_dir, 'deep3dlayout_mesh_intersect_1210',
                           f'{args.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name="deep3dlayout")
    path = os.path.join(args.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(args)))

    tb_writer = SummaryWriter(log_dir=LOG_DIR)

    mesh_criterion = MeshLoss(chamfer_weight=args.dp3d_position_loss_coef, normal_weight=args.dp3d_normal_loss_coef,
                              edge_weight=args.dp3d_edge_loss_coef, gt_num_samples=5000,
                              pred_num_samples=5000)
    sharp_criterion = L_sharp_loss
    smooth_criterion = L_smooth_loss

    if(args.optimizer == 'adam'):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.layout_learning_rate,
                                weight_decay=args.weight_decay)
    elif(args.optimizer == 'adamW'):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.layout_learning_rate,
                                weight_decay=args.weight_decay)
    else:
        print("unkown optimizer!")


    scheduler = get_scheduler(optimizer, len(train_loader), args)


    for epoch in range(args.start_epoch, args.max_epoch + 1):

        tic = time.time()

        train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, mesh_criterion, sharp_criterion, smooth_criterion, optimizer, scheduler, args)

        if(len(optimizer.param_groups)>1):
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}, lr_decoder {:.5f}'.format(epoch, (time.time() - tic),
                                                                   optimizer.param_groups[0]['lr'],
                                                                   optimizer.param_groups[1]['lr']))
        else:
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}'.format(epoch, (time.time() - tic),optimizer.param_groups[0]['lr']))

        if(epoch%args.val_freq==0):
            evaluate_one_epoch(epoch//args.val_freq, test_loader, model, mesh_criterion, sharp_criterion, smooth_criterion, args)
            if(epoch%100):
                save_checkpoint(args, epoch, model, optimizer, scheduler, save_best=True)

    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))

