import torch
import numpy as np
from module.layout.model.hohonet import HoHoNet
import os

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    # dataloder
    igibson_root_dir = '/mnt/workspace/code/PanoHolisticUnderstanding/igibson_vote_data_242'
    layout_pretrain_ckpt = '/Users/yuandong/Downloads/ckpt_epoch_last.pth'
    Network = HoHoNet(pretrain=layout_pretrain_ckpt)
    exit()

    if(not os.path.exists(layout_pretrain_ckpt)):
        print('can not find ',layout_pretrain_ckpt)
        exit()
    output_dir = 'layout_test_output'
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

    Network = HoHoNet(pretrain=layout_pretrain_ckpt)
    for batch_idx, batch_data_label in enumerate(test_loader):
        for key in batch_data_label:
            try:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
            except:
                print("Can Not Load: ",key)
        inputs = {'image': batch_data_label['image']}

        cor_id = Network.infer(inputs)['cor_id']

        fname = batch_data_label['scan_name'][0]
        with open(os.path.join(output_dir, f'{fname}.txt'), 'w') as f:
            for u, v in cor_id:
                f.write(f'{u:.1f} {v:.1f}\n')



