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
from module.deep3dlayout.mesh_loss import MeshLoss
from module.deep3dlayout.custom_losses.losses import L_sharp_loss, L_smooth_loss
from pytorch3d.io import save_obj
from utils.custom_sharder import CustomShader
from utils.softargmax import softargmax1d
import math
Num_heading_bin = 12
import trimesh
from pytorch3d.structures import Meshes
from shapely.geometry import Polygon
from pytorch3d.transforms import RotateAxisAngle,Translate
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)

def get_standard_bbox(device = 'cuda'):

    transformation = np.eye(4)
    transformation[2, 3] = -0.5
    corners = np.array([[-0.5,-0.5],[-0.5,0.5],[0.5,0.5],[0.5,-0.5]])
    polygon = Polygon(corners.tolist())
    mesh = trimesh.creation.extrude_polygon(polygon, height=1, transform=transformation)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces_verts_idx = torch.tensor(mesh.faces)
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_rgb)
    textured_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_verts_idx.to(device)],
        textures=textures.to(device)
    )
    return textured_mesh

def physical_violation_loss(end_points, renderer, image_size):
    prefixes = ['layout_refine_']

    for prefix in prefixes:
        bs = end_points[f'{prefix}center'].shape[0]

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        pred_heading_residual =  end_points[f'{prefix}heading_residuals']
        heading_scores = end_points[f'{prefix}heading_scores']

        pred_heading_class_argmax = torch.argmax(heading_scores, -1) # B,num_proposal
        pred_heading_residual = torch.gather(pred_heading_residual, 2, pred_heading_class_argmax.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = softargmax1d(heading_scores.contiguous().view(-1,Num_heading_bin)).view(bs,-1).unsqueeze(-1)

        pred_heading_gap = 2 * math.pi / float(Num_heading_bin)
        pred_angle = (pred_heading_gap * pred_heading_class + pred_heading_residual).squeeze(2)

        bbox_num = bs * 256
        batch_bbox_mesh = get_standard_bbox()
        batch_bbox_mesh = batch_bbox_mesh.extend(bbox_num)

        # apply_size
        all_size = pred_size.contiguous().view(-1,3)
        batch_bbox_mesh = batch_bbox_mesh.scale_verts(all_size)

        # apply_rotation
        mesh_rotation = RotateAxisAngle(pred_angle.contiguous().view(-1), degrees=False, axis="Z")
        # apply_translation
        mesh_translation = Translate(pred_center.contiguous().view(-1,3))
        transform = mesh_rotation.compose(mesh_translation)
        updated_verts = transform.transform_points(batch_bbox_mesh.verts_padded())
        batch_bbox_mesh = batch_bbox_mesh.update_padded(updated_verts)

        renderer_bbox_image = renderer(batch_bbox_mesh)
        renderer_bbox_image = renderer_bbox_image.view(bs,-1,image_size, image_size,1)

        # render layout_mesh
        pred_mesh = end_points['deep3d_meshes'][-1]

        verts_rgb = torch.ones_like(pred_mesh.verts_padded())
        textures = TexturesVertex(verts_rgb.to('cuda'))
        textured_mesh = Meshes(
            verts= pred_mesh.verts_padded(),
            faces=pred_mesh.faces_padded(),
            textures=textures
        )

        renderer_layout_image = renderer(textured_mesh).unsqueeze(1)
        outside_mask = renderer_layout_image<0.5
        outside_bbox_image = outside_mask * renderer_bbox_image
        outside_bbox_image_1d = outside_bbox_image.view(bs,256,-1)
        inters_sum = torch.sum(outside_bbox_image_1d,2)

        obj_ass_mask = end_points[f'{prefix}objectness_label'].float()

        mask_and = torch.sum(torch.logical_and(renderer_layout_image>0.5, renderer_bbox_image>0.5).view(bs,256,-1),2)
        renderer_bbox_image_mask = torch.sum((renderer_bbox_image>0.5).view(bs,256,-1),2)
        inside_mask = (mask_and==renderer_bbox_image_mask)

        inv_mask_and = torch.sum(torch.logical_and(renderer_layout_image<0.5, renderer_bbox_image>0.5).view(bs,256,-1),2)
        outside_mask = (inv_mask_and==renderer_bbox_image_mask)

        inside_outside_mask = ~torch.logical_or(inside_mask,outside_mask)

        final_mask = inside_outside_mask * obj_ass_mask

        physical_loss = torch.sum(inters_sum * final_mask)/(torch.sum(final_mask) + 1e-6)/100

        end_points[f'{prefix}physical_loss'] = physical_loss

        return physical_loss, end_points

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'igibson':
        from igibson.igbson_detection_dataloader_nodepth import IGbsonDetectionDataset
        from igibson.model_util_igbson import IGbsonDatasetConfig

        DATASET_CONFIG = IGbsonDatasetConfig()
        TRAIN_DATASET = IGbsonDetectionDataset('train', num_points=args.num_point,
                                                 augment=True,
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

def train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion_obj, criterion_layout, criterion_depth, optimizer, scheduler, config):
    model.train()  # set model to training mode
    stat_dict = {}
    layout_stat_dict = {}
    depth_stat_dict = 0

    # render bbox
    image_size = 160
    R, T = look_at_view_transform(10, 0, 0)
    cameras_up = FoVOrthographicCameras(scale_xyz=((0.12, 0.12, 0.12),), device='cuda', R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer_up = MeshRasterizer(
        cameras=cameras_up,
        raster_settings=raster_settings
    )
    shader_up = CustomShader(device='cuda', cameras=cameras_up)
    renderer_up = MeshRenderer(rasterizer_up, shader_up)

    R, T = look_at_view_transform(10, 90, 0)
    cameras_left = FoVOrthographicCameras(scale_xyz=((0.12, 0.12, 0.12),), device='cuda', R=R, T=T)
    rasterizer_left = MeshRasterizer(
        cameras=cameras_left,
        raster_settings=raster_settings
    )
    shader_left = CustomShader(device='cuda', cameras=cameras_left)
    renderer_left = MeshRenderer(rasterizer_left, shader_left)
    vert_criterion = torch.nn.MSELoss()

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

        if config.layout_flag and config.depth_flag:
            # layout-loss
            gt_meshes = (batch_data_label['points_gt'], batch_data_label['normals_gt'])
            mesh_loss, layout_losses = criterion_layout(meshes_pred=[end_points['deep3d_meshes'][1],end_points['deep3d_meshes'][2]], meshes_gt=gt_meshes)

            # compute sharp-loss
            # sharp_loss_0 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][0], meshes_gt=batch_data_label)
            sharp_loss_1 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][1], meshes_gt=batch_data_label)
            sharp_loss_2 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][2], meshes_gt=batch_data_label)

            # layout_losses['sharp_loss_0'] = sharp_loss_0.item()
            layout_losses['sharp_loss_1'] = sharp_loss_1.item()
            layout_losses['sharp_loss_2'] = sharp_loss_2.item()

            # smooth_loss_0 = L_smooth_loss(end_points['deep3d_meshes'][0])
            smooth_loss_1 = L_smooth_loss(end_points['deep3d_meshes'][1])
            smooth_loss_2 = L_smooth_loss(end_points['deep3d_meshes'][2])

            # layout_losses['smooth_loss_0'] = smooth_loss_0.item()
            layout_losses['smooth_loss_1'] = smooth_loss_1.item()
            layout_losses['smooth_loss_2'] = smooth_loss_2.item()

            init_vert_loss = vert_criterion(end_points['deep3d_meshes'][0].verts_padded(),
                                            batch_data_label['initial_mesh_loc'])
            layout_losses['init_vert_loss'] = init_vert_loss.item()
            init_vert_loss_coef = 1.0

            mesh_loss_all = mesh_loss +  (sharp_loss_1 + sharp_loss_2) / 2.0 * config.dp3d_sharp_loss_coef + \
                         (smooth_loss_1 + smooth_loss_2) / 2.0 * config.dp3d_smooth_loss_coef + init_vert_loss * init_vert_loss_coef

            end_points['layout_loss'] = mesh_loss_all
            violation_loss, end_points = physical_violation_loss(end_points, renderer_up, image_size)

            depth_loss = criterion_depth(batch_data_label['gt_depth'], end_points['pred_depth'])
            end_points['depth_loss'] = depth_loss

            end_points['total_loss'] = end_points['object_loss'] + mesh_loss_all + violation_loss * config.phy_viol_delta + \
                                       depth_loss * config.depth_delta

            depth_stat_dict = depth_stat_dict + depth_loss.item()


            for key in layout_losses:
                if key not in layout_stat_dict: layout_stat_dict[key] = 0
                if isinstance(layout_losses[key], float):
                    layout_stat_dict[key] += layout_losses[key]
                else:
                    layout_stat_dict[key] += layout_losses[key].item()

        else:
            print("error!")
            exit()
            end_points['total_loss'] = end_points['object_loss']

        optimizer.zero_grad()
        end_points['total_loss'].backward()
        if config.clip_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        scheduler.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % config.print_freq == 0:
            if config.layout_flag:
                logger.info(f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ' + ''.join(
                    [f'{key} {layout_stat_dict[key] / (float(batch_idx + 1)):.4f} \t' for key in layout_stat_dict]))
            logger.info('grad_norm: {}'.format(grad_total_norm.item()))
            logger.info('Depth Loss: {}'.format(depth_stat_dict/float(batch_idx + 1)))
            logger.info(f'T[{time}] Train: [{batch_idx + 1}/{len(test_loader)}]  ' + ''.join(
                [f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                 for key in sorted(stat_dict.keys()) if 'loss' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if
                                 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'last_' in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
            for ihead in range(args.num_decoder_layers - 2, -1, -1):
                logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                     for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in end_points.items():
            if('loss' in k):
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), cur_iter)

        if config.layout_flag:
            for key in layout_losses:
                    k = 'train/%s' % key
                    tb_writer.add_scalar(k, layout_losses[key], cur_iter)

def evaluate_one_epoch(epoch, test_loader, model, criterion_obj, criterion_layout, config):

    model.eval()  # set model to training mode
    eval_loss = 0
    AP_IOU_THRESHOLDS = [0.15]

    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,
                   'dataset_config': DATASET_CONFIG}

    if config.num_decoder_layers > 0:
        if (config.layout_flag):
            prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)] + ['layout_refine_']
        else:
            prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
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

        if(batch_idx<2):
            for j in range(4):  # write a maxmimum of four images
                tb_writer.add_image("gt_depth/{}".format(j), batch_data_label["gt_depth"][j].data/batch_data_label["gt_depth"][j].data.max(), epoch*len(test_loader)+1+batch_idx)
                tb_writer.add_image("pred_depth/{}".format(j), end_points["pred_depth"][j].data/end_points["pred_depth"][j].data.max(), epoch*len(test_loader)+1+batch_idx)

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
            # layout-loss
            gt_meshes = (batch_data_label['points_gt'], batch_data_label['normals_gt'])
            mesh_loss, layout_losses = criterion_layout(meshes_pred=end_points['deep3d_meshes'], meshes_gt=gt_meshes)

            # compute sharp-loss
            sharp_loss_0 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][0], meshes_gt=batch_data_label)
            sharp_loss_1 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][1], meshes_gt=batch_data_label)
            sharp_loss_2 = L_sharp_loss(meshes_pred=end_points['deep3d_meshes'][2], meshes_gt=batch_data_label)

            layout_losses['sharp_loss_0'] = sharp_loss_0.item()
            layout_losses['sharp_loss_1'] = sharp_loss_1.item()
            layout_losses['sharp_loss_2'] = sharp_loss_2.item()

            smooth_loss_0 = L_smooth_loss(end_points['deep3d_meshes'][0])
            smooth_loss_1 = L_smooth_loss(end_points['deep3d_meshes'][1])
            smooth_loss_2 = L_smooth_loss(end_points['deep3d_meshes'][1])

            layout_losses['smooth_loss_0'] = smooth_loss_0.item()
            layout_losses['smooth_loss_1'] = smooth_loss_1.item()
            layout_losses['smooth_loss_2'] = smooth_loss_2.item()

            mesh_loss_all = mesh_loss +  (sharp_loss_0 + sharp_loss_1 + sharp_loss_2) / len(end_points['deep3d_meshes']) * config.dp3d_sharp_loss_coef + \
                         (smooth_loss_0 + smooth_loss_1 + smooth_loss_2) / len(end_points['deep3d_meshes']) * config.dp3d_smooth_loss_coef

            end_points['layout_loss'] = mesh_loss_all
            end_points['total_loss'] = end_points['object_loss'] + mesh_loss_all

            os.makedirs(os.path.join(LOG_DIR, 'dump_eval'), exist_ok=True)
            output_filepath = os.path.join(LOG_DIR, 'dump_eval', str(epoch) + "_" + str(batch_idx))

            if (batch_idx<5):
                save_obj(output_filepath+"_initial.obj", end_points['deep3d_meshes'][0].cpu().detach().verts_padded()[0,:,:], end_points['deep3d_meshes'][0].cpu().detach().faces_padded()[0,:,:])
                save_obj(output_filepath+"_refine.obj", end_points['deep3d_meshes'][1].cpu().detach().verts_padded()[0,:,:], end_points['deep3d_meshes'][1].cpu().detach().faces_padded()[0,:,:])
        else:
            end_points['total_loss'] = end_points['object_loss']

        eval_loss = eval_loss + end_points['total_loss'].item()

        cur_iter = (epoch-1)*len(train_loader)+batch_idx

        for k, v in end_points.items():
            if('loss' in k):
                k = 'val/%s' % k
                tb_writer.add_scalar(k, v.item(), cur_iter)

        if config.layout_flag:
            for key in layout_losses:
                    k = 'val/%s' % key
                    tb_writer.add_scalar(k, layout_losses[key], cur_iter)

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
    from models.detector_layout_mesh_depth import GroupFreeDetectorDeep3D
    num_input_channel = int(args.use_color) * 3

    model = GroupFreeDetectorDeep3D(num_class=DATASET_CONFIG.num_class,
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
                              layout_flag=config.layout_flag,
                              depth_flag=config.depth_flag)

    return model


if __name__ == '__main__':
    args = parse_option()

    train_loader, test_loader, DATASET_CONFIG = get_loader(args)

    model = get_model(config = args, DATASET_CONFIG = DATASET_CONFIG)

    if(torch.cuda.is_available()):
        model = model.cuda()
    if(args.layout_flag):
        output_name = 'layoutmesh_detection_depth_1view_1212_surp'
    else:
        output_name = 'detection_depth_1view_1212_surp'
    LOG_DIR = os.path.join(args.log_dir, output_name, f'{args.dataset}_{int(time.time())}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(args.log_dir, output_name, f'{args.dataset}_{int(time.time())}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name=output_name)
    path = os.path.join(args.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))
    logger.info(str(vars(args)))

    tb_writer = SummaryWriter(log_dir=LOG_DIR)

    criterion_obj = get_loss
    criterion_layout = MeshLoss(chamfer_weight=args.dp3d_position_loss_coef, normal_weight=args.dp3d_normal_loss_coef,
                              edge_weight=args.dp3d_edge_loss_coef, gt_num_samples=5000,
                              pred_num_samples=5000)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad and "layout_estimation" not in n and "depth_depth_estimation" not in n]},
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad and "layout_estimation" not in n and "depth_depth_estimation" not in n],
            "lr": args.decoder_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad and "layout_estimation" in n and "depth_depth_estimation" not in n],
            "lr": args.layout_learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if "decoder" not in n and p.requires_grad and "layout_estimation" not in n and "depth_depth_estimation" in n],
            "lr": args.depth_learning_rate,
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
    from module.depth_estimation.losses import BerhuLoss
    criterion_depth = BerhuLoss()
    for epoch in range(args.start_epoch, args.max_epoch + 1):

        tic = time.time()

        train_one_epoch(epoch, train_loader, DATASET_CONFIG, model, criterion_obj, criterion_layout, criterion_depth, optimizer, scheduler, args)
        if(len(optimizer.param_groups)>1):
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}, lr_decoder {:.5f}, lr_layout {:.5f},'.format(epoch, (time.time() - tic),
                                                                   optimizer.param_groups[0]['lr'],
                                                                   optimizer.param_groups[1]['lr'],
                                                                   optimizer.param_groups[2]['lr']))
        else:
            logger.info('epoch {}, total time {:.2f}, '
                        'lr_base {:.5f}'.format(epoch, (time.time() - tic),optimizer.param_groups[0]['lr']))

        if(epoch%args.val_freq==0 or epoch==1):
            evaluate_one_epoch(epoch//args.val_freq, test_loader, model, criterion_obj, criterion_layout, args)
            save_checkpoint(args, epoch, model, optimizer, scheduler, save_best=True)

    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))

