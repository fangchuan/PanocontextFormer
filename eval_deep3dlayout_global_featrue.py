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


def get_model():
    from module.deep3dlayout.deep3dlayout_model_global_featrue import Deep3DlayoutNet

    model = Deep3DlayoutNet(backbone='resnet18', decoder_type='rcnn_p2m_mhsa_pos_dual', full_size=True, hidden_dim=128)

    return model

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
    checkpoint_path = "log/deep3dlayout_global_0107/igibson_1673061685/ckpt_epoch_last.pth"
    args.log_dir = os.path.dirname(checkpoint_path)
    args.method_name = 'deep3dlayout'
    test_loder = get_loader(args)
    model = get_model()

    LOG_DIR = os.path.join(args.log_dir, args.method_name+"_dump", f'{time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(args.log_dir, args.method_name+"_dump", f'{time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())}')
    args.log_dir = LOG_DIR
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(output=args.log_dir, name=args.method_name)

    model = load_checkpoint(checkpoint_path, model)
    if(torch.cuda.is_available()):
        model = model.cuda()
    model.eval()

    stat_dict = {}
    chamfer = []
    F1_score = []
    F3_score = []
    F5_score = []
    for batch_idx, batch_data_label in tqdm(enumerate(test_loder)):
        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == "scan_name"): continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        with torch.no_grad():
            end_points = model(batch_data_label)

        gt_meshes = Meshes(verts=batch_data_label['gt_mesh_vertics'], faces=batch_data_label['gt_mesh_faces'])
        cur_metrics = compare_meshes(end_points['deep3d_meshes'][-1], gt_meshes, scale=1.0, reduce=False)
        logger.info("Chamfer-L2: {}".format(cur_metrics["Chamfer-L2"][0].item()))
        chamfer.append(cur_metrics["Chamfer-L2"][0].item())
        F1_score.append(cur_metrics["F1@0.100000"][0].item())
        F3_score.append(cur_metrics["F1@0.300000"][0].item())
        F5_score.append(cur_metrics["F1@0.500000"][0].item())

        # save-mesh
        output_filepath = os.path.join(LOG_DIR,batch_data_label["scan_name"][0])
        save_obj(output_filepath+"_gt.obj", batch_data_label['gt_mesh_vertics'].squeeze(), batch_data_label['gt_mesh_faces'].squeeze())
        save_obj(output_filepath+"_initial.obj", end_points['deep3d_meshes'][0].cpu().detach().verts_packed(), end_points['deep3d_meshes'][0].cpu().detach().faces_packed())
        save_obj(output_filepath+"_refine.obj", end_points['deep3d_meshes'][1].cpu().detach().verts_packed(), end_points['deep3d_meshes'][1].cpu().detach().faces_packed())
    logger.info("************* Average CD-Value: {}".format(np.mean(np.array(chamfer))))
    logger.info("************* Average F1-Score: {}".format(np.mean(np.array(F1_score))))
    logger.info("************* Average F3-Score: {}".format(np.mean(np.array(F3_score))))
    logger.info("************* Average F5-Score: {}".format(np.mean(np.array(F5_score))))
