import os
from pdb import Pdb
import random
import imageio
import numpy as np
import torch.utils.data as data

from data import common

from utils import interact
import PIL.Image


class Dataset(data.Dataset):
    """Basic dataloader class
    """
    def __init__(self, args, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode

        self.modes = ()
        self.set_modes()
        self._check_mode()

        self.set_keys()

        if self.mode == 'train':
            dataset = args.data_train
        elif self.mode == 'val':
            dataset = args.data_val
        elif self.mode == 'test':
            dataset = args.data_test
        elif self.mode == 'demo':
            pass
        else:
            raise NotImplementedError('not implemented for this mode: {}!'.format(self.mode))

        if self.mode == 'demo':
            self.subset_root = args.demo_input_dir
        else:
            # self.subset_root = os.path.join(args.data_root, dataset, self.mode)
            if args.split == 1:
                if self.mode == 'train':
                    dataset = ['indoor_flying_2', 'indoor_flying_3']
                else:
                    dataset = 'indoor_flying_1'
            elif args.split == 3:
                if self.mode == 'train':
                    dataset = ['indoor_flying_1', 'indoor_flying_2']
                else:
                    dataset = 'indoor_flying_3'
            
            if self.mode == 'train':
                self.subset_root = []
                for set in dataset:
                    self.subset_root.append(os.path.join(args.data_root, set))
            else:
                self.subset_root = os.path.join(args.data_root, dataset)

            
        # import pdb
        # pdb.set_trace()

        # FRAMES_FILTER_FOR_TEST = {
        # 'indoor_flying': {
        # 1: list(range(140, 1201)),
        # 2: list(range(120, 1421)),
        # 3: list(range(73, 1616)),
        # 4: list(range(190, 290))
        # }
        # }

        # FRAMES_FILTER_FOR_TRAINING = {
        # 'indoor_flying': {
        #     1: list(range(80, 1260)),
        #     2: list(range(160, 1580)),
        #     3: list(range(125, 1815)),
        #     4: list(range(190, 290))
        # }
        # }

        self.left_image_list = []
        self.right_image_list = []
        self.left_event_list = []
        self.right_event_list = []
        self.disp_list = []



        self._scan()


    def set_modes(self):
        self.modes = ('train', 'val', 'test', 'demo')

    def _check_mode(self):
        """Should be called in the child class __init__() after super
        """
 
        if self.mode not in self.modes:
            raise NotImplementedError('mode error: not for {}'.format(self.mode))

        return

    def set_keys(self):

        self.left_image_key = 'image0'
        self.right_image_key = 'image1'
        # self.left_event_key = 'select0'
        # self.right_event_key = 'select1'
        # self.left_event_key = 'numvox0'
        # self.right_event_key = 'numvox1'
        self.left_event_key = 'voxel0_orig'
        self.right_event_key = 'voxel1_orig'
        self.disp_key = 'disparity_image'


        self.non_left_image_keys = []
        self.non_right_image_keys = []
        self.non_left_event_keys = []
        self.non_right_event_keys = []
        self.non_disp_keys = []

        return

    def _scan(self, root=None):
        """Should be called in the child class __init__() after super
        """
        if root is None:
            root = self.subset_root

        # if self.blur_key in self.non_blur_keys:
        #     self.non_blur_keys.remove(self.blur_key)
        # if self.sharp_key in self.non_sharp_keys:
        #     self.non_sharp_keys.remove(self.sharp_key)
        # if self.event_key in self.non_event_keys:
        #     self.non_event_keys.remove(self.event_key)

        if self.left_image_key in self.non_left_image_keys:
            self.non_left_image_keys.remove(self.left_image_key)
        if self.right_image_key in self.non_right_image_keys:
            self.non_right_image_keys.remove(self.right_image_key)
        if self.left_event_key in self.non_left_event_keys:
            self.non_left_event_keys.remove(self.left_event_key)
        if self.right_event_key in self.non_right_event_keys:
            self.non_right_event_keys.remove(self.right_event_key)
        if self.disp_key in self.non_disp_keys:
            self.non_disp_keys.remove(self.disp_key)
        


        def _key_check(path, true_key, false_keys):
            path = os.path.join(path, '')
            if path.find(true_key) >= 0:
                for false_key in false_keys:
                    if path.find(false_key) >= 0:
                        return False

                return True
            else:
                return False
        # FRAMES_FILTER_FOR_TEST = {
        # 'indoor_flying': {
        # 1: list(range(140, 1201)),
        # 2: list(range(120, 1421)),
        # 3: list(range(73, 1616)),
        # 4: list(range(190, 290))
        # }
        # }

        # FRAMES_FILTER_FOR_TRAINING = {
        # 'indoor_flying': {
        #     1: list(range(80, 1260)),
        #     2: list(range(160, 1580)),
        #     3: list(range(125, 1815)),
        #     4: list(range(190, 290))
        # }
        # }

        # original
        FILTER_TEST = {
        # 'indoor_flying_1': list(range(140, 1201)),
        # 'indoor_flying_2': list(range(120, 1421)),
        # 'indoor_flying_3': list(range(73, 1616))
        # 'indoor_flying_1': list(range(140, 1001)),
        'indoor_flying_1': list(range(140, 1001)),
        'indoor_flying_2': list(range(120, 1221)),
        'indoor_flying_3': list(range(273, 1616))
        }

        # test
        # FILTER_TEST = {
        # # 'indoor_flying_1': list(range(140, 1201)),
        # # 'indoor_flying_2': list(range(120, 1421)),
        # # 'indoor_flying_3': list(range(73, 1616))
        # 'indoor_flying_1': list(range(140, 161)),
        # 'indoor_flying_2': list(range(120, 141)),
        # 'indoor_flying_3': list(range(73, 93))
        # }

        # original
        FILTER_TRAIN = {
        'indoor_flying_1': list(range(80, 1260)),
        'indoor_flying_2': list(range(160, 1580)),
        'indoor_flying_3': list(range(125, 1815))
        }
        
        # test
        # FILTER_TRAIN = {
        # 'indoor_flying_1': list(range(80, 90)),
        # 'indoor_flying_2': list(range(160, 180)),
        # 'indoor_flying_3': list(range(125, 145))
        # }

        def _get_list_by_key(root, true_key, false_keys):
            data_list = []
            if isinstance(root, (list, tuple)):
                for rt in root:
                    for sub, dirs, files in os.walk(rt):
                        if not dirs:
                            file_list = [os.path.join(sub, f) for f in files if int(f.split('.')[0]) in FILTER_TRAIN[rt.split('/')[-1]]]
                            if _key_check(sub, true_key, false_keys):
                                data_list += file_list
            else:
                for sub, dirs, files in os.walk(root):
                    if not dirs:
                        file_list = [os.path.join(sub, f) for f in files if int(f.split('.')[0]) in FILTER_TEST[root.split('/')[-1]]]
                        if _key_check(sub, true_key, false_keys):
                            data_list += file_list

            
            data_list.sort()
            return data_list

        def _rectify_keys():

            self.left_image_key = os.path.join(self.left_image_key, '')
            self.non_left_image_keys = [os.path.join(non_left_image_key, '') for non_left_image_key in self.non_left_image_keys]
            self.left_event_key = os.path.join(self.left_event_key, '')
            self.non_left_event_keys = [os.path.join(non_left_event_key, '') for non_left_event_key in self.non_left_event_keys]
            self.right_image_key = os.path.join(self.right_image_key, '')
            self.non_right_image_keys = [os.path.join(non_right_image_key, '') for non_right_image_key in self.non_right_image_keys]
            self.right_event_key = os.path.join(self.right_event_key, '')
            self.non_right_event_keys = [os.path.join(non_right_event_key, '') for non_right_event_key in self.non_right_event_keys]
            self.disp_key = os.path.join(self.disp_key, '')
            self.non_disp_keys = [os.path.join(non_disp_key, '') for non_disp_key in self.non_disp_keys]
            

        _rectify_keys()
        
        self.left_image_list = _get_list_by_key(root, self.left_image_key, self.non_left_image_keys)
        self.left_event_list = _get_list_by_key(root, self.left_event_key, self.non_left_event_keys)
        self.right_image_list = _get_list_by_key(root, self.right_image_key, self.non_right_image_keys)
        self.right_event_list = _get_list_by_key(root, self.right_event_key, self.non_right_event_keys)
        self.disp_list = _get_list_by_key(root, self.disp_key, self.non_disp_keys)
        
        
        if len(self.left_image_list) > 0:
            assert(len(self.left_image_list) == len(self.left_event_list))
        if len(self.right_image_list) > 0:
            assert(len(self.right_image_list) == len(self.right_event_list))
        if len(self.left_image_list) > 0:
            assert(len(self.left_image_list) == len(self.right_image_list))
        if len(self.disp_list) > 0:
            assert(len(self.disp_list) == len(self.left_image_list))

        return

    def __getitem__(self, idx):

        left_image = imageio.imread(self.left_image_list[idx], pilmode='RGB')
        right_image = imageio.imread(self.right_image_list[idx], pilmode='RGB')
        imgs = [left_image, right_image]

        left_event = np.load(self.left_event_list[idx])
        right_event = np.load(self.right_event_list[idx])
        # print(left_event.shape)

        disp = np.array(PIL.Image.open(self.disp_list[idx])).astype(np.uint8)
        invalid_disparity = (disp == 255.0)
        disparity_image = (disp / 7.0)
        disparity_image[invalid_disparity] = float('inf')



        pad_width = 0   # dummy value
        # if self.mode == 'train':
        #     # imgs, left_event, right_event = common.crop_with_event(*imgs, left_event = left_event, right_event = right_event, ps=self.args.patch_size)
        #     imgs[0], pad_width = common.pad(imgs[0], divisor=64)
        # elif self.mode == 'demo':
        #     imgs[0], pad_width = common.pad(imgs[0], divisor=2**(self.args.n_scales-1))   # pad in case of non-divisible size
        # else:
        #     # imgs[0], pad_width = common.pad(imgs[0], divisor=2**(self.args.n_scales-1))
        #     # event, pad_width = common.pad(event, divisor=2**(self.args.n_scales-1))
        #     pass    # deliver test image as is.

        ## padding
        # imgs[0], pad_width = common.pad(imgs[0], divisor=64)
        # imgs[1], pad_width = common.pad(imgs[1], divisor=64)
        # left_event, _ = common.event_pad(left_event, divisor=64)
        # right_event, _ = common.event_pad(right_event, divisor=64)

        # print(imgs[0].shape)
        # print(imgs[1].shape)
        # print(left_event.shape)
        # print(right_event.shape)
        # import pdb
        # pdb.set_trace()

        noise_imgs = [imgs[0], imgs[1]]
        noise_imgs[0] = common.add_noise(imgs[0], sigma_sigma=2, rgb_range=self.args.rgb_range)
        noise_imgs[1] = common.add_noise(imgs[1], sigma_sigma=2, rgb_range=self.args.rgb_range)        
        
    
        # print(event.shape)
        
        if self.args.gaussian_pyramid:
            if self.mode == ('train' or 'demo'):

                imgs = common.generate_pyramid(*imgs, n_scales=self.args.n_scales)
                left_event = common.generate_event_pyramid(left_event, n_scales=self.args.n_scales)
                right_event = common.generate_event_pyramid(right_event, n_scales=self.args.n_scales)
            else:
                # left_event, pad_width = common.event_pad(left_event, divisor=2**(self.args.n_scales-1))
                # right_event, pad_width = common.event_pad(right_event, divisor=2**(self.args.n_scales-1))
                # imgs[0], pad_width = common.pad(imgs[0], divisor=2**(self.args.n_scales-1))
                # imgs[1], pad_width = common.pad(imgs[1], divisor=2**(self.args.n_scales-1))
                imgs = common.generate_pyramid(*imgs, n_scales=self.args.n_scales)
                left_event = common.generate_event_pyramid(*left_event, n_scales=self.args.n_scales)
                right_event = common.generate_event_pyramid(*right_event, n_scales=self.args.n_scales)
             

        imgs = common.np2tensor(*imgs)
        noise_imgs = common.np2tensor(*noise_imgs)

        left_event = common.event2tensor(left_event)[0]
        right_event = common.event2tensor(right_event)[0]

        if self.mode == 'train':
            relpath = os.path.relpath(self.left_image_list[idx], self.subset_root[0])
        else:
            relpath = os.path.relpath(self.left_image_list[idx], self.subset_root)

        
        # blur = imgs[0]
        
        left_img = imgs[0]
        right_img = imgs[1]
        # sharp = imgs[1] if len(imgs) > 1 else False
        left_noise = noise_imgs[0]
        right_noise = noise_imgs[1]


        return left_img, right_img, pad_width, idx, relpath, left_event, right_event, left_noise, right_noise, disparity_image

    def __len__(self):
        return len(self.left_image_list)
        # return 32





