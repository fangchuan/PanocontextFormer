# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os
import sys

from utils import pc_util

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

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

def dump_results(end_points, dump_dir, config, inference_switch=False, prefix = 'last_'):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
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
    if(f'{prefix}size_scores' in end_points):
        pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
    else:
        pred_size = end_points[f'{prefix}pred_size'].squeeze(2).detach().cpu().numpy()

    # OTHERS
    pred_mask = end_points[f'{prefix}pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
        objectness_prob = sigmoid(obj_logits)[i, :, 0]  # (B,K)

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(seed_xyz[i,:,:], os.path.join(dump_dir, '%06d_seed_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(pred_center[i,:,0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            pc_util.write_ply(pred_center[i,objectness_prob>DUMP_CONF_THRESH,0:3], os.path.join(dump_dir, '%06d_confident_proposal_pc.ply'%(idx_beg+i)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                if (f'{prefix}size_scores' in end_points):
                    obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                    pred_size_class[i,j], pred_size_residual[i,j])
                else:
                    obb = config.param2obb_direct(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                    pred_size[i,j])
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))

                pred_conf_bbox = []
                pred_conf_nms_bbox = []
                pred_nms_bbox = []
                for bbox_i in range(obbs.shape[0]):
                    if(objectness_prob[bbox_i]>DUMP_CONF_THRESH):
                        pred_conf_bbox.append(obbs[bbox_i])
                    if(pred_mask[i,bbox_i]==1):
                        pred_nms_bbox.append(obbs[bbox_i])
                    if(objectness_prob[bbox_i]>DUMP_CONF_THRESH and pred_mask[i,bbox_i]==1):
                        pred_conf_nms_bbox.append(obbs[bbox_i])

                pc_util.write_oriented_bbox(np.array(pred_conf_bbox), os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(np.array(pred_conf_nms_bbox), os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                pc_util.write_oriented_bbox(np.array(pred_nms_bbox), os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))
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
    objectness_label = end_points[f'{prefix}objectness_label'].detach().cpu().numpy() # (B,K,)
    objectness_mask = end_points[f'{prefix}objectness_mask'].detach().cpu().numpy() # (B,K,)

    for i in range(batch_size):
        if np.sum(objectness_label[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_label[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply'%(idx_beg+i)))
        if np.sum(objectness_mask[i,:])>0:
            pc_util.write_ply(pred_center[i,objectness_mask[i,:]>0,0:3], os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply'%(idx_beg+i)))
        pc_util.write_ply(gt_center[i,:,0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply'%(idx_beg+i)))
        pc_util.write_ply_color(pred_center[i,:,0:3], objectness_label[i,:], os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.obj'%(idx_beg+i)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[i,j] == 0: continue
            obb = config.param2obb(gt_center[i,j,0:3], gt_heading_class[i,j], gt_heading_residual[i,j],
                            gt_size_class[i,j], gt_size_residual[i,j])
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply'%(idx_beg+i)))

    # OPTIONALL, also dump prediction and gt details
    if 'batch_pred_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_pred_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' '+str(t[2]))
                fout.write('\n')
            fout.close()
    if 'batch_gt_map_cls' in end_points:
        for ii in range(batch_size):
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt'%(ii)), 'w')
            for t in end_points['batch_gt_map_cls'][ii]:
                fout.write(str(t[0])+' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()
