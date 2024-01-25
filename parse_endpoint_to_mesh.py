from glob import glob
import os
import numpy as np
import torch
from igibson.model_util_igbson import IGbsonDatasetConfig
from utils.dump_helper import sigmoid
import utils.pc_util as pc_util
from igibson.example_data.auto_encoder import PointCloudAE

def generate_mesh_from_latend_code(estimator, latent_code):
    with torch.no_grad():
        _, shape_out = estimator(None, torch.FloatTensor(latent_code).unsqueeze(0))
        shape_out = shape_out.squeeze().detach().numpy()
    return shape_out

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat

if __name__ == '__main__':
    end_points = {}
    all_filename = glob(os.path.join("./endpoint/"+"*.npy"))
    for i in range(len(all_filename)):
        key = all_filename[i].split("/")[-1][:-4]
        try:
            end_points[key] = torch.from_numpy(np.load(all_filename[i],allow_pickle=True))
        except:
            print("can not load ",key)
            pass

    batch_size = end_points['scan_idx'].shape[0]
    config = IGbsonDatasetConfig()
    prefix = 'last_'
    dump_dir = './dump_dir'
    os.makedirs(dump_dir,exist_ok=True)

    model_path = "./igibson/example_data/ig_autoencoder.pth"
    estimator = PointCloudAE(128, 2048)
    estimator.load_state_dict(torch.load(model_path, map_location='cpu'))
    estimator.eval()


    objectness_scores = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()  # (B,K,2)
    pred_center = end_points[f'{prefix}center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
    pred_latent_code = end_points[f'{prefix}emb_codes']

    # save GT-BBox
    gt_center = end_points['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_mask = end_points['box_label_mask'].cpu().numpy()  # B,K2
    gt_heading_class = end_points['heading_class_label'].cpu().numpy()  # B,K2
    gt_heading_residual = end_points['heading_residual_label'].cpu().numpy()  # B,K2
    gt_size_class = end_points['size_class_label'].cpu().numpy()  # B,K2
    gt_size_residual = end_points['size_residual_label'].cpu().numpy()  # B,K2,3
    objectness_label = end_points[f'{prefix}objectness_label'].detach().cpu().numpy()  # (B,K,)
    objectness_mask = end_points[f'{prefix}objectness_mask'].detach().cpu().numpy()  # (B,K,)

    # OTHERS
    pred_mask = end_points[f'{prefix}pred_mask']  # B,num_proposal
    idx_beg = 0
    DUMP_CONF_THRESH = 0.5
    save_generate_pc = False
    if save_generate_pc:
        for i in range(batch_size):
            print("processing bach {}".format(i))

            obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
            objectness_prob = sigmoid(obj_logits)[i, :, 0]  # (B,K)

            # Dump predicted bounding boxes
            if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
                num_proposal = pred_center.shape[1]
                obbs = []
                for j in range(num_proposal):
                    obb = config.param2obb(pred_center[i,j,0:3], pred_heading_class[i,j], pred_heading_residual[i,j],
                                    pred_size_class[i,j], pred_size_residual[i,j])
                    obbs.append(obb)
                if len(obbs)>0:
                    obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)

                    pred_conf_bbox = []
                    pred_conf_nms_bbox = []
                    pred_nms_bbox = []
                    merged_obj_pc = []
                    for bbox_i in range(obbs.shape[0]):
                        if(objectness_prob[bbox_i]>DUMP_CONF_THRESH):
                            pred_conf_bbox.append(obbs[bbox_i])
                        if(pred_mask[i,bbox_i]==1):
                            pred_nms_bbox.append(obbs[bbox_i])
                        if(objectness_prob[bbox_i]>DUMP_CONF_THRESH and pred_mask[i,bbox_i]==1):
                            pred_conf_nms_bbox.append(obbs[bbox_i])
                            rotation_matrix = heading2rotmat(-config.class2angle(pred_heading_class[i,bbox_i], pred_heading_residual[i,bbox_i]))
                            translation = pred_center[i,bbox_i,0:3]
                            size = config.class2size(int(pred_size_class[i,bbox_i]), pred_size_residual[i,bbox_i])
                            shape_out = generate_mesh_from_latend_code(estimator, pred_latent_code[i,bbox_i])
                            obj_vertices = (rotation_matrix @ (shape_out * size).T).T + translation
                            merged_obj_pc.append(obj_vertices)

                    # pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_pred_bbox.ply'%(idx_beg+i)))
                    # pc_util.write_oriented_bbox(np.array(pred_conf_bbox), os.path.join(dump_dir, '%06d_pred_confident_bbox.ply'%(idx_beg+i)))
                    pc_util.write_oriented_bbox(np.array(pred_conf_nms_bbox), os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply'%(idx_beg+i)))
                    pc_util.write_ply(np.concatenate(merged_obj_pc,0), os.path.join(dump_dir, '%06d_merged_obj.ply'%(idx_beg+i)))
                    # pc_util.write_oriented_bbox(np.array(pred_nms_bbox), os.path.join(dump_dir, '%06d_pred_nms_bbox.ply'%(idx_beg+i)))



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