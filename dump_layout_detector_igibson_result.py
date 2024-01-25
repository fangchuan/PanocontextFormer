import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from utils import pc_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

from utils import setup_logger
from models import GroupFreeDetector, get_loss
from models import APCalculator, parse_predictions, parse_groundtruths
from igibson.model_util_igbson import IGbsonDatasetConfig
import trimesh
from module.layout.misc.post_proc import np_coor2xy,np_coory2v

DC = IGbsonDatasetConfig()

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def sigmoid(x):
    ''' Numpy function for softmax'''
    s = 1 / (1 + np.exp(-x))
    return s

def heading2rotmat(heading_angle):
    pass
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat

def write_wireframe_bboxes(heading_class_label,heading_residual_label,center_label,
                           size_class_label,size_residual_label, sem_cls):

    x_corners = [-1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2]
    y_corners = [1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2]
    z_corners = [1 / 2, 1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2]
    shape_out = np.vstack([x_corners, y_corners, z_corners]).transpose()

    all_obj_vertices = []

    rotation_matrix = heading2rotmat(
        -DC.class2angle(int(heading_class_label), heading_residual_label))
    translation = center_label
    size = DC.class2size(int(size_class_label), size_residual_label)
    corners_3d = (rotation_matrix @ (shape_out * size).T).T + translation

    radius = 0.025
    mesh = []
    colorbox = DC.get_colorbox()
    layout_lines = [[0,1],[0,3],[0,4],[1,2],[1,5],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]

    for indexes in layout_lines:
        line = corners_3d[indexes]
        line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
        line_mesh.visual.vertex_colors[:] = np.append(colorbox[sem_cls], 255).astype(np.uint8)
        mesh.append(line_mesh)

    return mesh

def dump_results(end_points, dump_dir, config, inference_switch=False, prefix = 'last_', scan_name = ""):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    DUMP_CONF_THRESH = 0.5
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    # NETWORK OUTPUTS
    seed_xyz = end_points['seed_xyz'].detach().cpu().numpy() # (B,num_seed,3)

    objectness_scores = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points[f'{prefix}center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    # OTHERS
    pred_mask = end_points[f'{prefix}pred_mask'] # B,num_proposal
    pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'], -1)  # B,num_proposal

    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
        objectness_prob = sigmoid(obj_logits)[i, :, 0]  # (B,K)


        # Dump various point clouds
        # pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '{}_seed_pc.ply'.format(scan_name)))
        # pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            # pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '{}_confident_proposal_pc.ply'.format(scan_name)))
            pass

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            meshes = []
            for j in range(num_proposal):
                obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                pred_size_class[i,j], pred_size_residual[i,j])
                mesh = write_wireframe_bboxes(pred_heading_class[i,j], pred_heading_residual[i,j], pred_center[i,j,0:3],
                                              pred_size_class[i,j], pred_size_residual[i,j], pred_sem_cls[i,j])
                obbs.append(obb)
                meshes.append(mesh)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                # pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

                pred_conf_bbox = []
                pred_conf_nms_bbox = []
                pred_nms_bbox = []
                mesh_conf_nms = []
                mesh_conf_nms_out = []
                for bbox_i in range(obbs.shape[0]):
                    if(objectness_prob[bbox_i]>DUMP_CONF_THRESH):
                        pred_conf_bbox.append(obbs[bbox_i])
                    if(pred_mask[i,bbox_i]==1):
                        pred_nms_bbox.append(obbs[bbox_i])
                    if(objectness_prob[bbox_i]>DUMP_CONF_THRESH and pred_mask[i,bbox_i]==1):
                        pred_conf_nms_bbox.append(obbs[bbox_i])
                        mesh_conf_nms.append(meshes[bbox_i])

                for mesh_i in range(len(mesh_conf_nms)):
                    for mesh_sub in mesh_conf_nms[mesh_i]:
                        mesh_conf_nms_out.append(mesh_sub)
                # pc_util.write_oriented_bbox(np.array(pred_conf_bbox), os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(np.array(pred_conf_nms_bbox), os.path.join(dump_dir, '{}_pred_confident_nms_bbox.ply'.format(scan_name)))
                trimesh.exchange.export.export_mesh(sum(mesh_conf_nms_out), os.path.join(dump_dir, '{}_wireframe_bboxes.ply'.format(scan_name)))
                # pc_util.write_oriented_bbox(np.array(pred_nms_bbox), os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
                # pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH,:], os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                # pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                # pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    gt_center = end_points['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy() # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy() # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy() # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy() # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy() # B,K2,3
    # objectness_label = end_points[f'{prefix}objectness_label'].detach().cpu().numpy() # (B,K,)
    # objectness_mask = end_points[f'{prefix}objectness_mask'].detach().cpu().numpy() # (B,K,)

    for i in range(batch_size):
        # if np.sum(objectness_label[i,:])>0:
            # pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
            # pass
        # if np.sum(objectness_mask[i,:])>0:
            # pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
            # pass
        # pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        # pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '{}_gt_bbox.ply'.format(scan_name)))

def infer(end_points = None, prefixe = ''):
    from module.layout.misc.post_proc import np_refine_by_fix_z, gen_ww, infer_coory
    from scipy.ndimage.filters import maximum_filter
    from shapely.geometry import Polygon

    pred_bon = end_points[prefixe+'bon'].clone()
    pred_cor = end_points[prefixe+'cor'].clone()
    H, W = 512, 1024

    y_bon_ = (pred_bon[0].cpu().numpy() / np.pi + 0.5) * H - 0.5
    y_cor_ = pred_cor[0, 0].sigmoid().cpu().numpy()
    # Init floor/ceil plane
    z0 = 50
    _, z1 = np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    def find_N_peaks(signal, r, min_v, N):
        max_v = maximum_filter(signal, size=r, mode='wrap')
        pk_loc = np.where(max_v == signal)[0]
        pk_loc = pk_loc[signal[pk_loc] > min_v]
        if N is not None:
            order = np.argsort(-signal[pk_loc])
            pk_loc = pk_loc[order[:N]]
            pk_loc = pk_loc[np.argsort(pk_loc)]
        return pk_loc, signal[pk_loc]

    post_force_cuboid = False
    min_v = 0 if post_force_cuboid else 0.05
    r = int(round(W * 0.05 / 2))
    N = 4 if post_force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6),
                                   force_cuboid=post_force_cuboid)
    if not post_force_cuboid:
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            import sys
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j * 2] = cor[j, 0], cor[j, 1]
        cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]
    return {'cor_id': cor_id, 'y_bon_': y_bon_, 'y_cor_': y_cor_}

def dump_layout(end_points, dump_dir, scan_name = ""):
    W = 1024
    H = 512
    cor = infer(end_points=end_points,prefixe='initial_')['cor_id']
    N = len(cor) // 2
    floor_z = -1.6
    floor_xy = np_coor2xy(cor[1::2], floor_z, W, H, floorW=1, floorH=1)
    c = np.sqrt((floor_xy ** 2).sum(1))
    v = np_coory2v(cor[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    wf_points = [[-x, -y, floor_z] for x, y in floor_xy] + \
                [[-x, -y, ceil_z] for x, y in floor_xy]
    wf_points = np.array(wf_points)
    wf_lines = [[i, (i + 1) % N] for i in range(N)] + \
               [[i + N, (i + 1) % N + N] for i in range(N)] + \
               [[i, i + N] for i in range(N)]
    mesh = []
    for indexes in wf_lines:
        line = wf_points[indexes]
        line_mesh = trimesh.creation.cylinder(0.025, sections=8, segment=line)
        mesh.append(line_mesh)
    mesh = sum(mesh)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(dump_dir, '{}_layout_wireframe.ply'.format(scan_name)))

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'igibson':
        from igibson.igbson_detection_dataloader import IGbsonDetectionDataset
        from igibson.model_util_igbson import IGbsonDatasetConfig

        DATASET_CONFIG = IGbsonDatasetConfig()
        TEST_DATASET = IGbsonDetectionDataset('test', num_points=args.num_point,
                                            augment=False,
                                            use_color=True if args.use_color else False,
                                            use_height=True if args.use_height else False,
                                            use_v1=(not args.use_sunrgbd_v2),
                                            ROOT_DIR=args.igibson_root_dir,
                                            latent_code_dim=args.emb_dim)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    logger.info(str(len(TEST_DATASET)))

    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1,
                                 shuffle=False,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn)
    return TEST_DATALOADER, DATASET_CONFIG


def get_model(config, DATASET_CONFIG):
    from models.detector_layout import GroupFreeDetectorHorizonNet
    num_input_channel = int(config.use_color) * 3

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


def load_checkpoint(args, model):
    # Load checkpoint if there is any
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        len_key = len("module.")
        import collections
        new_dict = collections.OrderedDict()
        for k in list(state_dict.keys()):
            if(k[:len_key] == "module."):
                new_dict[k[len_key:]] = state_dict[k]
            else:
                new_dict[k] = state_dict[k]
        model.load_state_dict(new_dict, strict=True)
        print(f"{args.checkpoint_path} loaded successfully!!!")
    else:
        raise FileNotFoundError
    return model


def evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, args, time=0):
    stat_dict = {}
    if args.num_decoder_layers > 0:
        if args.dataset == 'igibson':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            _prefixes += ['layout_refine_']
            prefixes = _prefixes.copy() + ['last_three_'] + ['all_layers_']
    else:
        prefixes = ['proposal_']  # only proposal
        _prefixes = prefixes

    if args.num_decoder_layers >= 3:
        last_three_prefixes = ['last_', f'{args.num_decoder_layers - 2}head_', f'{args.num_decoder_layers - 3}head_']
    elif args.num_decoder_layers == 2:
        last_three_prefixes = ['last_', '0head_']
    elif args.num_decoder_layers == 1:
        last_three_prefixes = ['last_']
    else:
        last_three_prefixes = []

    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)

    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    for batch_idx, batch_data_label in enumerate(test_loader):
        save_name = 'Beechwood_1_int_00001'
        if(batch_data_label['scan_name'][0]!=save_name):
            continue

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

        # Compute loss
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=args.num_decoder_layers,
                                     query_points_generator_loss_coef=args.query_points_generator_loss_coef,
                                     obj_loss_coef=args.obj_loss_coef,
                                     box_loss_coef=args.box_loss_coef,
                                     sem_cls_loss_coef=args.sem_cls_loss_coef,
                                     query_points_obj_topk=args.query_points_obj_topk,
                                     center_loss_type=args.center_loss_type,
                                     center_delta=args.center_delta,
                                     size_loss_type=args.size_loss_type,
                                     size_delta=args.size_delta,
                                     heading_loss_type=args.heading_loss_type,
                                     heading_delta=args.heading_delta,
                                     size_cls_agnostic=args.size_cls_agnostic,
                                     emb_code_delta = args.emb_code_delta)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        for prefix in prefixes:
            if prefix == 'last_three_':
                end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                           for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                   for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in last_three_prefixes], 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = torch.cat([end_points[f'{ppx}pred_size']
                                                                  for ppx in last_three_prefixes], 1)
                else:
                    end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                                    for ppx in last_three_prefixes], 1)
                    end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                       for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in last_three_prefixes], 1)

            elif prefix == 'all_layers_':
                end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                           for ppx in _prefixes], 1)
                end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                   for ppx in _prefixes], 1)
                end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in _prefixes], 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = torch.cat([end_points[f'{ppx}pred_size']
                                                                  for ppx in _prefixes], 1)
                else:
                    end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                                    for ppx in _prefixes], 1)
                    end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                       for ppx in _prefixes], 1)
                end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in _prefixes], 1)
                end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in _prefixes], 1)

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix,
                                                   size_cls_agnostic=args.size_cls_agnostic)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT,
                                                  size_cls_agnostic=args.size_cls_agnostic)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        if(True):
            output_dir = './endpoint_'+ save_name
            os.makedirs(output_dir, exist_ok=True)
            for key in end_points:
                output_filename = os.path.join(output_dir, key + ".npy")
                try:
                    if isinstance(end_points[key],float) or isinstance(end_points[key],np.ndarray):
                        output_obj = end_points[key]
                    else:
                        output_obj = end_points[key].to('cpu').numpy()
                    np.save(output_filename, output_obj)
                except:
                    output_obj_list = end_points[key]
                    print("can not save ",key)
            exit()
        else:
            dump_results(end_points, opt.dump_dir, DATASET_CONFIG, prefix='layout_refine_',scan_name=batch_data_label['scan_name'][0])
            if(args.layout_flag):
                dump_layout(end_points, opt.dump_dir, scan_name=batch_data_label['scan_name'][0])

        if (batch_idx + 1) % 10 == 0:
            logger.info(f'T[{time}] Eval: [{batch_idx + 1}/{len(test_loader)}]  ' + ''.join(
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

    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            logger.info(f'===================>T{time} {prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<==================')
            for key in metrics_dict:
                logger.info(f'{key} {metrics_dict[key]}')

            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()
    for mAP in mAPs:
        logger.info(f'T[{time}] IoU[{mAP[0]}]: ' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))
    return mAPs


def eval(args, avg_times=5):
    test_loader, DATASET_CONFIG = get_loader(args)
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model = get_model(args, DATASET_CONFIG)
    logger.info(str(model))
    model = load_checkpoint(args, model)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': DATASET_CONFIG}

    logger.info(str(datetime.now()))
    mAPs_times = [None for i in range(avg_times)]
    criterion_object = get_loss
    for i in range(avg_times):
        np.random.seed(i + args.rng_seed)
        mAPs = evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds,
                                 model, criterion_object, args, i)
        mAPs_times[i] = mAPs

    mAPs_avg = mAPs.copy()

    for i, mAP in enumerate(mAPs_avg):
        for key in mAP[1].keys():
            avg = 0
            for t in range(avg_times):
                cur = mAPs_times[t][i][1][key]
                avg += cur
            avg /= avg_times
            mAP[1][key] = avg

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \n' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \n' for key in sorted(mAP[1].keys())]))

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \t' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))


if __name__ == '__main__':
    from get_options import parse_option
    opt = parse_option()
    opt.checkpoint_path = "log/layout_detection/igibson_1667878570_example/best_valid_epoch_400.pth"
    opt.avg_times = 1
    checkout_dir = os.path.split(opt.checkpoint_path)[0]

    opt.dump_dir = os.path.join(checkout_dir, opt.dump_dir, f'dump_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    logger = setup_logger(output=opt.dump_dir, name="eval")

    eval(opt, opt.avg_times)
