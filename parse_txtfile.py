import cv2
import numpy as np


if __name__ == '__main__':
    txt_filepath = "/Users/yuandong/Downloads/log_eval_gp3d2.txt"
    print_list = ['eval chair Average Precision',
                    'eval sofa Average Precision',
                    'eval table Average Precision',
                    'eval fridge Average Precision',
                    'eval sink Average Precision',
                    'eval door Average Precision',
                    'eval floor_lamp Average Precision',
                    'eval bottom_cabinet Average Precision',
                    'eval top_cabinet Average Precision',
                    'eval sofa_chair Average Precision',
                    'eval dryer Average Precision']
    print_list_gp3d = ['eval INFO: chair Average Precision',
                    'eval INFO: sofa Average Precision',
                    'eval INFO: table Average Precision',
                    'eval INFO: fridge Average Precision',
                    'eval INFO: sink Average Precision',
                    'eval INFO: door Average Precision',
                    'eval INFO: floor_lamp Average Precision',
                    'eval INFO: bottom_cabinet Average Precision',
                    'eval INFO: top_cabinet Average Precision',
                    'eval INFO: sofa_chair Average Precision',
                    'eval INFO: dryer Average Precision']
    all_ap = []
    with open(txt_filepath,"r") as f:
        line_list = []
        for line in f.readlines():
            line = line.split('\n')
            line_list.append(line[0])

        for print_name in print_list_gp3d:
            for current_line in line_list:
                if (print_name in current_line):
                    print(current_line)
                    split_file = current_line.split(print_name)
                    all_ap.append(float(split_file[1]))

    mean_ap = np.mean(np.array(all_ap))
    print("mean ap(11): ",mean_ap)
