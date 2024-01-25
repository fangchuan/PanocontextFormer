import json
import numpy as np
import torch
import os
import time
from tqdm import tqdm

from get_options import parse_option
from utils.logger import setup_logger
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
import open3d as o3d
from utils.metrics import compare_meshes

def get_loader(config):
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    # Create Dataset and Dataloader

    from module.deep3dlayout.dataset_layoutmesh import PanoLayoutMeshDataset
    if(torch.cuda.is_available()):
        TEST_DATASET = PanoLayoutMeshDataset(root_dir='/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242', split = 'test')
    else:
        TEST_DATASET = PanoLayoutMeshDataset(root_dir='/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data', split = 'test')


    test_loder = torch.utils.data.DataLoader(TEST_DATASET,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=config.num_workers if torch.cuda.is_available() else 0,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               drop_last=False)


    print(f"test_loder_len: {len(test_loder)}")

    return test_loder


def load_checkpoint(checkpoint_path, model):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    logger.info("loading ... {}".format(checkpoint_path))

    return model

if __name__ == '__main__':
    args = parse_option()
    pred_mesh_dir = "log/panocontextformer_0114_F2/igibson_1673696403/dump/eval_igibson_1673833249_14863401/layout_mesh"
    args.log_dir = pred_mesh_dir
    args.method_name = 'deep3dlayout'
    test_loder = get_loader(args)
    LOG_DIR = args.log_dir

    logger = setup_logger(output=args.log_dir, name=args.method_name)

    stat_dict = {}
    chamfer = []
    F1_score = []
    F3_score = []
    F5_score = []
    for batch_idx, batch_data_label in tqdm(enumerate(test_loder)):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if key == "scan_name": continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        # mesh_filepath = os.path.join(pred_mesh_dir,batch_data_label['scan_name'][0]+"_cross.obj")
        mesh_filepath = os.path.join(pred_mesh_dir,batch_data_label['scan_name'][0]+"_refine.obj")
        # mesh_filepath = os.path.join(pred_mesh_dir,str(batch_idx)+"_refine.obj")
        verts, faces, aux = load_obj(mesh_filepath)
        faces_verts_idx = faces.verts_idx

        pred_mesh = Meshes(verts=[verts.cuda()], faces=[faces_verts_idx.cuda()])
        gt_meshes = Meshes(verts=batch_data_label['gt_mesh_vertics'], faces=batch_data_label['gt_mesh_faces'])

        cur_metrics = compare_meshes(pred_mesh, gt_meshes, scale=1.0, reduce=False)
        logger.info("Chamfer-L2: {}".format(cur_metrics["Chamfer-L2"][0].item()))
        chamfer.append(cur_metrics["Chamfer-L2"][0].item())
        F1_score.append(cur_metrics["F1@0.100000"][0].item())
        F3_score.append(cur_metrics["F1@0.300000"][0].item())
        F5_score.append(cur_metrics["F1@0.500000"][0].item())

        # save-mesh
        output_filepath = os.path.join(LOG_DIR,batch_data_label['scan_name'][0])
        save_obj(output_filepath+"_gt.obj", batch_data_label['gt_mesh_vertics'].squeeze(), batch_data_label['gt_mesh_faces'].squeeze())
    logger.info("************* Average CD-Value: {}".format(np.mean(np.array(chamfer))))
    logger.info("************* Average F1-Score: {}".format(np.mean(np.array(F1_score))))
    logger.info("************* Average F3-Score: {}".format(np.mean(np.array(F3_score))))
    logger.info("************* Average F5-Score: {}".format(np.mean(np.array(F5_score))))
