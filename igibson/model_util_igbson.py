# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class IGbsonDatasetConfig(object):
    def __init__(self):
        self.num_class = 57
        self.num_heading_bin = 12
        self.num_size_cluster = 57

        self.type2class={'basket':0, 'bathtub':1, 'bed':2, 'bench':3, 'bottom_cabinet':4,
                        'bottom_cabinet_no_top':5, 'carpet':6, 'chair':7, 'chest':8,
                        'coffee_machine':9, 'coffee_table':10, 'console_table':11,
                        'cooktop':12, 'counter':13, 'crib':14, 'cushion':15, 'dishwasher':16,
                        'door':17, 'dryer':18, 'fence':19, 'floor_lamp':20, 'fridge':21,
                        'grandfather_clock':22, 'guitar':23, 'heater':24, 'laptop':25,
                        'loudspeaker':26, 'microwave':27, 'mirror':28, 'monitor':29,
                        'office_chair':30, 'oven':31, 'piano':32, 'picture':33, 'plant':34,
                        'pool_table':35, 'range_hood':36, 'shelf':37, 'shower':38, 'sink':39,
                        'sofa':40, 'sofa_chair':41, 'speaker_system':42, 'standing_tv':43,
                        'stool':44, 'stove':45, 'table':46, 'table_lamp':47, 'toilet':48,
                        'top_cabinet':49, 'towel_rack':50, 'trash_can':51, 'treadmill':52,
                        'wall_clock':53, 'wall_mounted_tv':54, 'washer':55, 'window':56}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'basket':0, 'bathtub':1, 'bed':2, 'bench':3, 'bottom_cabinet':4,
                                'bottom_cabinet_no_top':5, 'carpet':6, 'chair':7, 'chest':8,
                                'coffee_machine':9, 'coffee_table':10, 'console_table':11,
                                'cooktop':12, 'counter':13, 'crib':14, 'cushion':15, 'dishwasher':16,
                                'door':17, 'dryer':18, 'fence':19, 'floor_lamp':20, 'fridge':21,
                                'grandfather_clock':22, 'guitar':23, 'heater':24, 'laptop':25,
                                'loudspeaker':26, 'microwave':27, 'mirror':28, 'monitor':29,
                                'office_chair':30, 'oven':31, 'piano':32, 'picture':33, 'plant':34,
                                'pool_table':35, 'range_hood':36, 'shelf':37, 'shower':38, 'sink':39,
                                'sofa':40, 'sofa_chair':41, 'speaker_system':42, 'standing_tv':43,
                                'stool':44, 'stove':45, 'table':46, 'table_lamp':47, 'toilet':48,
                                'top_cabinet':49, 'towel_rack':50, 'trash_can':51, 'treadmill':52,
                                'wall_clock':53, 'wall_mounted_tv':54, 'washer':55, 'window':56}

        self.eval_class = {
            'basket', 'bathtub', 'bed', 'bottom_cabinet', 'bottom_cabinet_no_top', 'carpet', 'chair', 'coffee_table',
            'counter', 'dishwasher', 'door', 'dryer', 'floor_lamp', 'fridge', 'heater', 'mirror', 'oven', 'piano',
            'picture', 'plant', 'range_hood', 'shelf', 'shower', 'sink', 'sofa', 'sofa_chair', 'stool', 'stove',
            'table', 'table_lamp', 'toilet', 'top_cabinet', 'towel_rack', 'trash_can', 'wall_mounted_tv', 'washer',
            'window'
        }

        self.type_mean_size = {'basket': np.array([ 0.9537104964256287 , 0.5851879715919495 , 0.7986712455749512 ])/2,
                                'bathtub': np.array([ 1.9644783735275269 , 1.2292708158493042 , 0.4795311689376831 ])/2,
                                'bed': np.array([ 1.6841776371002197 , 2.1207566261291504 , 0.9023095965385437 ])/2,
                                'bench': np.array([ 0.9537104964256287 , 0.5851879715919495 , 0.7986712455749512 ])/2,
                                'bottom_cabinet': np.array([ 0.9382281303405762 , 0.4766995310783386 , 0.8935611248016357 ])/2,
                                'bottom_cabinet_no_top': np.array([ 0.6620856523513794 , 0.5809850096702576 , 0.860061526298523 ])/2,
                                'carpet': np.array([ 2.151893138885498 , 1.7289674282073975 , 0.008340648375451565 ])/2,
                                'chair': np.array([ 0.49972373247146606 , 0.485862135887146 , 0.8833140134811401 ])/2,
                                'chest': np.array([ 0.839425265789032 , 0.4480343461036682 , 0.4610443413257599 ])/2,
                                'coffee_machine': np.array([ 0.3199999928474426 , 0.5100001692771912 , 0.39500999450683594 ])/2,
                                'coffee_table': np.array([ 1.0849802494049072 , 0.610287606716156 , 0.3741244375705719 ])/2,
                                'console_table': np.array([ 0.5393643975257874 , 0.30293288826942444 , 0.7016563415527344 ])/2,
                                'cooktop': np.array([ 0.9022448062896729 , 0.5661224722862244 , 0.0662694200873375 ])/2,
                                'counter': np.array([ 1.3481099605560303 , 0.7256325483322144 , 0.031498733907938004 ])/2,
                                'crib': np.array([ 1.3499994277954102 , 0.7700002789497375 , 1.1286001205444336 ])/2,
                                'cushion': np.array([ 0.559999942779541 , 0.5 , 0.29700008034706116 ])/2,
                                'dishwasher': np.array([ 0.615109920501709 , 0.5887311100959778 , 0.720461905002594 ])/2,
                                'door': np.array([ 1.153671383857727 , 0.1349860280752182 , 2.052752733230591 ])/2,
                                'dryer': np.array([ 0.7798706889152527 , 0.8126306533813477 , 1.1706041097640991 ])/2,
                                'fence': np.array([ 2.525038242340088 , 0.06290895491838455 , 0.9900005459785461 ])/2,
                                'floor_lamp': np.array([ 0.3431675434112549 , 0.3658372461795807 , 1.5414444208145142 ])/2,
                                'fridge': np.array([ 0.9032005071640015 , 0.8046241402626038 , 1.7146306037902832 ])/2,
                                'grandfather_clock': np.array([ 0.654999852180481 , 0.34499993920326233 , 2.0789995193481445 ])/2,
                                'guitar': np.array([ 0.4424999952316284 , 0.42750000953674316 , 1.039500117301941 ])/2,
                                'heater': np.array([ 0.9537104964256287 , 0.5851879715919495 , 0.7986712455749512 ])/2,
                                'laptop': np.array([ 0.3110658526420593 , 0.22855591773986816 , 0.009900000877678394 ])/2,
                                'loudspeaker': np.array([ 0.4118577539920807 , 0.14886096119880676 , 0.23760013282299042 ])/2,
                                'microwave': np.array([ 0.7647744417190552 , 0.4859127402305603 , 0.36950355768203735 ])/2,
                                'mirror': np.array([ 1.2192249298095703 , 0.1699540913105011 , 0.9136002659797668 ])/2,
                                'monitor': np.array([ 0.7091198563575745 , 0.17492295801639557 , 0.39500999450683594 ])/2,
                                'office_chair': np.array([ 0.4677187502384186 , 0.4371362626552582 , 0.8304939866065979 ])/2,
                                'oven': np.array([ 0.8152874708175659 , 0.6419697999954224 , 0.8876680731773376 ])/2,
                                'piano': np.array([ 1.399999737739563 , 0.4699999988079071 , 1.1879998445510864 ])/2,
                                'picture': np.array([ 0.7122608423233032 , 0.020000284537672997 , 0.6293489933013916 ])/2,
                                'plant': np.array([ 0.3740699589252472 , 0.35362833738327026 , 0.9977086782455444 ])/2,
                                'pool_table': np.array([ 2.5898988246917725 , 1.330068826675415 , 0.6732000112533569 ])/2,
                                'range_hood': np.array([ 1.0384379625320435 , 0.6340109705924988 , 0.3464999496936798 ])/2,
                                'shelf': np.array([ 1.3647478818893433 , 0.38001877069473267 , 1.0892810821533203 ])/2,
                                'shower': np.array([ 1.2218611240386963 , 1.2968895435333252 , 2.1901416778564453 ])/2,
                                'sink': np.array([ 0.802036464214325 , 0.583819568157196 , 0.9459133744239807 ])/2,
                                'sofa': np.array([ 2.2969467639923096 , 1.1115189790725708 , 0.7260889410972595 ])/2,
                                'sofa_chair': np.array([ 0.8503119945526123 , 0.8457909822463989 , 0.8222343325614929 ])/2,
                                'speaker_system': np.array([ 0.23528534173965454 , 0.42803260684013367 , 0.4949999451637268 ])/2,
                                'standing_tv': np.array([ 0.9700977802276611 , 0.16601000726222992 , 0.6506500840187073 ])/2,
                                'stool': np.array([ 0.7444347143173218 , 0.7035099267959595 , 0.43735018372535706 ])/2,
                                'stove': np.array([ 0.9892788529396057 , 0.671051025390625 , 0.9405001997947693 ])/2,
                                'table': np.array([ 1.1392592191696167 , 0.6618147492408752 , 0.7233097553253174 ])/2,
                                'table_lamp': np.array([ 0.2823675572872162 , 0.26674267649650574 , 0.5805978775024414 ])/2,
                                'toilet': np.array([ 0.5000061988830566 , 0.7036129832267761 , 0.7112369537353516 ])/2,
                                'top_cabinet': np.array([ 0.8434890508651733 , 0.38297489285469055 , 0.7246139049530029 ])/2,
                                'towel_rack': np.array([ 0.6693085432052612 , 0.06796742975711823 , 0.0381857231259346 ])/2,
                                'trash_can': np.array([ 0.24549008905887604 , 0.22235290706157684 , 0.4270589053630829 ])/2,
                                'treadmill': np.array([ 0.85999995470047 , 1.7700001001358032 , 1.3859999179840088 ])/2,
                                'wall_clock': np.array([ 0.9537104964256287 , 0.5851879715919495 , 0.7986712455749512 ])/2,
                                'wall_mounted_tv': np.array([ 1.0971304178237915 , 0.11266380548477173 , 0.5831246376037598 ])/2,
                                'washer': np.array([ 0.7049159407615662 , 0.7734740972518921 , 1.1879997253417969 ])/2,
                                'window': np.array([ 1.6537036895751953 , 0.17392657697200775 , 1.4010463953018188 ])/2}

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb
    
    def param2obb_direct(self, center, heading_class, heading_residual, box_size):
        heading_angle = self.class2angle(heading_class, heading_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb
    
    def get_colorbox(self):
        import seaborn as sns
        IG56CLASSES = [
            'basket', 'bathtub', 'bed', 'bench', 'bottom_cabinet',
            'bottom_cabinet_no_top', 'carpet', 'chair', 'chest',
            'coffee_machine', 'coffee_table', 'console_table',
            'cooktop', 'counter', 'crib', 'cushion', 'dishwasher',
            'door', 'dryer', 'fence', 'floor_lamp', 'fridge',
            'grandfather_clock', 'guitar', 'heater', 'laptop',
            'loudspeaker', 'microwave', 'mirror', 'monitor',
            'office_chair', 'oven', 'piano', 'picture', 'plant',
            'pool_table', 'range_hood', 'shelf', 'shower', 'sink',
            'sofa', 'sofa_chair', 'speaker_system', 'standing_tv',
            'stool', 'stove', 'table', 'table_lamp', 'toilet',
            'top_cabinet', 'towel_rack', 'trash_can', 'treadmill',
            'wall_clock', 'wall_mounted_tv', 'washer', 'window'
        ]

        IG59CLASSES = IG56CLASSES + ['walls', 'floors', 'ceilings']
        igibson_colorbox = np.array(sns.hls_palette(n_colors=len(IG59CLASSES), l=.45, s=.8))
        return igibson_colorbox*255

if __name__ == '__main__':
    DATASET_CONFIG = IGbsonDatasetConfig()
