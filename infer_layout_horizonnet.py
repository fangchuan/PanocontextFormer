import torch
import numpy as np
from module.horizonnet.layout_estimation import HorizonNet,HorizonNetUpSample
import os
from tqdm import tqdm


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

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

if __name__ == '__main__':
    # dataloder
    if (torch.cuda.is_available()):
        igibson_root_dir = '/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242'
    else:
        igibson_root_dir = '/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data'
    layout_pretrain_ckpt = '/mnt/workspace/code/DeepPanoContext/out/layout_estimation/21022217101943/model_best.pth'

    if(not os.path.exists(layout_pretrain_ckpt)):
        print('can not find ',layout_pretrain_ckpt)
        exit()
    else:
        print('loading: ',layout_pretrain_ckpt)
    output_dir = 'layout_test_output'
    os.system('rm -rf layout_test_output')
    os.makedirs(output_dir,exist_ok=True)

    from igibson.igbson_detection_dataloader import IGbsonDetectionDataset
    from igibson.model_util_igbson import IGbsonDatasetConfig

    DATASET_CONFIG = IGbsonDatasetConfig()
    TEST_DATASET = IGbsonDetectionDataset('test', num_points=50000,
                                           augment=False,
                                           use_color=True,
                                           use_height=False,
                                           use_v1=False,
                                           ROOT_DIR=igibson_root_dir,
                                           latent_code_dim=512,
                                           layout_dataset_flag=True)

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)

    print('TEST_DATASET length: ',len(test_loader))

    Network = HorizonNet(cfg=None, pretrain_ckpt=layout_pretrain_ckpt)
    if (torch.cuda.is_available()):
        Network = Network.cuda()
    Network.eval()
    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):

        if(torch.cuda.is_available()):
            for key in batch_data_label:
                if(key == 'scan_name'):
                    continue
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        with torch.no_grad():
            end_points = Network(batch_data_label['image'])

        cor_id = infer(end_points=end_points,prefixe='initial_')['cor_id']

        fname = batch_data_label['scan_name'][0]
        with open(os.path.join(output_dir, f'{fname}.txt'), 'w') as f:
            for u, v in cor_id:
                f.write(f'{u:.1f} {v:.1f}\n')
                
    os.system('python eval_layout.py')



