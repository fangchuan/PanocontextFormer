import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
)

from utils.custom_sharder import CustomShader
from utils import setup_logger
from models import GroupFreeDetector, get_loss
from models import APCalculator, parse_predictions, parse_groundtruths
from pytorch3d.io import save_obj
from train_detection_layoutmesh_joint_loss import physical_violation_loss

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
    from models.detector_layout_mesh import GroupFreeDetectorDeep3D
    num_input_channel = int(config.use_color) * 3

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
            if(args.layout_flag):
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
    
    # render bbox
    image_size = 160
    R, T = look_at_view_transform(10, 0, 0)
    cameras = FoVOrthographicCameras(scale_xyz=((0.12, 0.12, 0.12),), device='cuda', R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    shader = CustomShader(device='cuda', cameras=cameras)
    renderer = MeshRenderer(rasterizer, shader)

    for batch_idx, batch_data_label in enumerate(test_loader):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        end_points = model(batch_data_label)
        if(args.layout_flag):
            os.makedirs(os.path.join(opt.dump_dir,'layout_mesh'),exist_ok=True)

            output_filepath = os.path.join(opt.dump_dir,'layout_mesh',batch_data_label['scan_name'][0])
            save_obj(output_filepath + "_initial.obj", end_points['deep3d_meshes'][0].cpu().detach().verts_padded()[0, :, :],
                     end_points['deep3d_meshes'][0].cpu().detach().faces_padded()[0, :, :])
            save_obj(output_filepath + "_refine.obj",
                     end_points['deep3d_meshes'][1].cpu().detach().verts_padded()[0, :, :],
                     end_points['deep3d_meshes'][1].cpu().detach().faces_padded()[0, :, :])
            if(len(end_points['deep3d_meshes'])>2):
                save_obj(output_filepath + "_cross.obj",
                         end_points['deep3d_meshes'][2].cpu().detach().verts_padded()[0, :, :],
                         end_points['deep3d_meshes'][2].cpu().detach().faces_padded()[0, :, :])

        for key in batch_data_label:
            if (key == 'scan_name'):
                continue
            end_points[key] = batch_data_label[key]
        
        # violation_loss, end_points = physical_violation_loss(end_points, renderer, image_size)

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
            if prefix == 'last_':
                os.makedirs(opt.dump_dir+"/map_cls", exist_ok=True)
                if prefix == 'last_':
                    if 'batch_pred_map_cls' in end_points:
                        for ii in range(1):
                            pred_map_cls = np.array(end_points['batch_pred_map_cls'][ii])
                            np.save(opt.dump_dir+"/map_cls"+'/'+batch_data_label['scan_name'][0]+"_pred_map_cls.npy",pred_map_cls)

                    if 'batch_gt_map_cls' in end_points:
                        for ii in range(1):
                            gt_map_cls = end_points['batch_gt_map_cls'][ii]
                            np.save(opt.dump_dir + "/map_cls" + '/' + batch_data_label['scan_name'][0] + "_gt_map_cls.npy", gt_map_cls)
            
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        if(args.save_end_point):
            output_dir = './render_dump'
            os.makedirs(output_dir, exist_ok=True)
            for key in end_points:
                output_filename = os.path.join(output_dir, key + ".npy")
                try:
                    if isinstance(end_points[key],float) or isinstance(end_points[key],np.ndarray):
                        output_obj = end_points[key]
                    else:
                        output_obj = end_points[key].detach().to('cpu').numpy()
                    np.save(output_filename, output_obj)
                except:
                    print("can not save: ",key)
                    continue
                    output_obj_list = end_points[key]
                    for idx in range(len(output_obj_list)):
                        current_obj = np.array(output_obj_list[idx])
                        output_filename = os.path.join(output_dir, key + str(idx)+ ".npy")
                        np.save(output_filename, current_obj)
                    pass
            exit()
        else:
            from utils.dump_helper import dump_results
            if batch_idx == 0:
                dump_results(end_points, opt.dump_dir, DATASET_CONFIG)

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
    opt.checkpoint_path = "log/panocontextformer_0114_F2/igibson_1673696403/ckpt_epoch_last.pth"
    opt.avg_times = 1
    opt.save_end_point = False
    
    checkout_dir = os.path.split(opt.checkpoint_path)[0]

    opt.dump_dir = os.path.join(checkout_dir, opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    logger = setup_logger(output=opt.dump_dir, name="eval")

    eval(opt, opt.avg_times)
