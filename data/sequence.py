from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from .representations import VoxelGrid
from .eventslicer import EventSlicer
import PIL.Image
from data import common
import os
import yaml





class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=5):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.image_height = 1080
        self.image_width = 1440

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        if self.mode == 'test':
            seq_str = str(seq_path).split('/')[-1]
            load_timestamps = np.loadtxt(seq_path / (seq_str + '.csv'), delimiter=",", dtype='int64')
            self.timestamps = load_timestamps[:, 0]
            self.image_index = load_timestamps[:, 1]
        else:
            # load disparity timestamps
            disp_dir = seq_path / 'disparity'
            assert disp_dir.is_dir()
            time_dir = seq_path / 'images'
            assert time_dir.is_dir()
            # self.timestamps = np.loadtxt(time_dir / 'timestamps.txt', dtype='int64')[::2]
            self.timestamps = np.loadtxt(time_dir / 'timestamps.txt', dtype='int64')[1::2]
            # self.timestamps[1::2]
        

            # load disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                disp_gt_pathstrings.append(str(entry))
            disp_gt_pathstrings.sort()
            self.disp_gt_pathstrings = disp_gt_pathstrings

            # assert len(self.disp_gt_pathstrings) == self.timestamps.size


        with open(seq_path / 'calibration/cam_to_cam.yaml') as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

        cam0_int = self.conf['intrinsics']['camRect0']['camera_matrix']
        cam1_int = self.conf['intrinsics']['camRect1']['camera_matrix']
        cam2_int = self.conf['intrinsics']['camRect2']['camera_matrix']
        cam3_int = self.conf['intrinsics']['camRect3']['camera_matrix']

        T32 = np.array(self.conf['extrinsics']['T_32'])
        T10 = np.array(self.conf['extrinsics']['T_10'])

        R_rect0 = np.array(self.conf['extrinsics']['R_rect0'])
        R_rect1 = np.array(self.conf['extrinsics']['R_rect1'])
        R_rect2 = np.array(self.conf['extrinsics']['R_rect2'])
        R_rect3 = np.array(self.conf['extrinsics']['R_rect3'])

        Kr0 = np.array([[cam0_int[0], 0, cam0_int[2]], 
                        [0, cam0_int[1], cam0_int[3]], 
                        [0, 0, 1]])
        Kr1 = np.array([[cam1_int[0], 0, cam1_int[2]], 
                        [0, cam1_int[1], cam1_int[3]], 
                        [0, 0, 1]])
        Kr2 = np.array([[cam2_int[0], 0, cam2_int[2]], 
                        [0, cam2_int[1], cam2_int[3]], 
                        [0, 0, 1]])
        Kr3 = np.array([[cam3_int[0], 0, cam3_int[2]], 
                        [0, cam3_int[1], cam3_int[3]], 
                        [0, 0, 1]])

        M1=np.matmul(Kr1,R_rect1)
        M2=np.matmul(M1,T10[:3,:3])
        M3=np.matmul(M2,np.linalg.inv(R_rect0))
        self.homography_left=np.matmul(M3,np.linalg.inv(Kr0))




        M1=np.matmul(Kr3,R_rect3)
        M2=np.matmul(M1,T32[:3,:3])
        M3=np.matmul(M2,np.linalg.inv(R_rect2))

        # M3 = Kr3
        self.homography_right=np.matmul(M3,np.linalg.inv(Kr2))

        # load image
        left_image_dir = seq_path / 'images/left/rectified'
        right_image_dir = seq_path / 'images/right/rectified'
        assert left_image_dir.is_dir()
        assert right_image_dir.is_dir()
        left_image_pathstrings = list()
        right_image_pathstrings = list()

        if self.mode == 'test':
            for entry in left_image_dir.iterdir():
                assert str(entry.name).endswith('.png')
            
                if int((entry.name).split('.')[0]) in self.image_index:
                    left_image_pathstrings.append(str(entry))
            left_image_pathstrings.sort()
            self.left_image_pathstrings = left_image_pathstrings

            for entry in right_image_dir.iterdir():
                assert str(entry.name).endswith('.png')
            
                if int((entry.name).split('.')[0]) in self.image_index:
                    right_image_pathstrings.append(str(entry))
            right_image_pathstrings.sort()
            self.right_image_pathstrings = right_image_pathstrings
        else:
            for entry in left_image_dir.iterdir():
                assert str(entry.name).endswith('.png')
            
                if int((entry.name).split('.')[0]) % 2 == 1:
                    left_image_pathstrings.append(str(entry))
            left_image_pathstrings.sort()
            self.left_image_pathstrings = left_image_pathstrings

            for entry in right_image_dir.iterdir():
                assert str(entry.name).endswith('.png')
            
                if int((entry.name).split('.')[0]) % 2 == 1:
                    right_image_pathstrings.append(str(entry))
            right_image_pathstrings.sort()
            self.right_image_pathstrings = right_image_pathstrings

            # assert len(self.left_image_pathstrings) == len(self.right_image_pathstrings)
            # assert len(self.left_image_pathstrings) == len(self.disp_gt_pathstrings)

            # Remove first disparity path and corresponding timestamp.
            # This is necessary because we do not have events before the first disparity map.
            assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
            self.disp_gt_pathstrings.pop(0)
            self.timestamps = self.timestamps[1:]


            self.left_image_pathstrings.pop(0)
            self.right_image_pathstrings.pop(0)

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]


        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)


        self.subset_root = os.path.join('../..', 'val')
        self.disp_key = 'disparity_image/'
        self.gt_disp_key = 'gt_disp/'
        self.left_image_key = 'image0'



    # def events_to_voxel_grid(self, x, y, p, t, device: str='cuda'):
    #     t = (t - t[0]).astype('float32')
    #     t = (t/t[-1])
    #     x = x.astype('float32')
    #     y = y.astype('float32')
    #     pol = p.astype('float32')
    #     return self.voxel_grid.convert(
    #             torch.from_numpy(x).to(device),
    #             torch.from_numpy(y).to(device),
    #             torch.from_numpy(pol).to(device),
    #             torch.from_numpy(t).to(device))

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        
        gt_disp = (np.array(disp_16bit).astype(np.float32))/256
        return gt_disp

        # @staticmethod
    def get_left_image(self, filepath: Path):
        assert filepath.is_file()   
        # image = imageio.imread(str(filepath), pilmode='RGB') 
        image = np.array(PIL.Image.open(str(filepath))).astype(np.uint8)
        warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography_left, (1440, 1080),  flags=cv2.WARP_INVERSE_MAP)
        warp_image=warp_image[0:480, 0:640,:]
        return warp_image

    # @staticmethod
    def get_right_image(self, filepath: Path):
        assert filepath.is_file()   
        # image = imageio.imread(str(filepath), pilmode='RGB') 
        image = np.array(PIL.Image.open(str(filepath))).astype(np.uint8)
        warp_image=cv2.warpPerspective(image[:,:,[2,1,0]], self.homography_right, (1440, 1080))
        warp_image=warp_image[0:480, 0:640,:]
        
        return warp_image

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        # return len(self.disp_gt_pathstrings)
        return len(self.timestamps)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us



        if self.mode == 'test':
            left_image_path = Path(self.left_image_pathstrings[index])
            right_image_path = Path(self.right_image_pathstrings[index])
            file_index = int(left_image_path.stem)
            
            output = {
                'file_index': file_index,
                'left_image': common.image2tensor(self.get_left_image(left_image_path)/255.0),
                'right_image': common.image2tensor(self.get_right_image(right_image_path)/255.0)
            }
            for location in self.locations:
                event_data = self.event_slicers[location].get_events(ts_start, ts_end)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                if 'representation' not in output:
                    output['representation'] = dict()
                output['representation'][location] = event_representation

        elif self.mode == 'train':
            disp_gt_path = Path(self.disp_gt_pathstrings[index])
            left_image_path = Path(self.left_image_pathstrings[index])
            right_image_path = Path(self.right_image_pathstrings[index])
            file_index = int(disp_gt_path.stem)

            disparity_gt = self.get_disparity_map(disp_gt_path)
            left_image = self.get_left_image(left_image_path)/255.0
            right_image = self.get_right_image(right_image_path)/255.0

    

            
            # py = random.randrange(0, 480-256+1)
            # px = random.randrange(0, 640-384+1)

            # left_image = common.crop(left_image, ps=[384,256], py=py, px=px)
            # right_image = common.crop(right_image, ps=[384,256], py=py, px=px)
            # disparity_gt = common.crop_disp(disparity_gt, ps=[384,256], py=py, px=px)

    
            vertical = 0
            # if np.random.random() < 0.5:
            #     vertical = 1
            #     left_image = np.copy(np.flipud(left_image))
            #     right_image = np.copy(np.flipud(right_image))

            #     disparity_gt = np.copy(np.flipud(disparity_gt))
                

            output = {
                'disparity_gt': disparity_gt,
                'file_index': file_index,
                'left_image': common.image2tensor(left_image),
                'right_image': common.image2tensor(right_image)
            }
            
            
            for location in self.locations:
                event_data = self.event_slicers[location].get_events(ts_start, ts_end)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                if 'representation' not in output:
                    output['representation'] = dict()
                # output['representation'][location] = common.crop_event(event_representation, ps=[384,256], py=py, px=px)
                # if vertical == 1:
                #     output['representation'][location] = torch.flipud(output['representation'][location])
                output['representation'][location] = event_representation

        else:
            disp_gt_path = Path(self.disp_gt_pathstrings[index])
            left_image_path = Path(self.left_image_pathstrings[index])
            right_image_path = Path(self.right_image_pathstrings[index])
            file_index = int(disp_gt_path.stem)

            disparity_gt = self.get_disparity_map(disp_gt_path)
            left_image = self.get_left_image(left_image_path)/255.0
            right_image = self.get_right_image(right_image_path)/255.0


            output = {
                'disparity_gt': disparity_gt,
                'file_index': file_index,
                'left_image': common.image2tensor(left_image),
                'right_image': common.image2tensor(right_image)
            }
            for location in self.locations:
                event_data = self.event_slicers[location].get_events(ts_start, ts_end)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                if 'representation' not in output:
                    output['representation'] = dict()
                output['representation'][location] = event_representation
        
        relpath = os.path.relpath(self.left_image_pathstrings[index], self.subset_root)
        # print(relpath)
        # print(str(relpath).split('images')[0])
        # import pdb; pdb.set_trace()
        left_image_path = relpath.replace('{}/'.format(self.left_image_key), '')
        # pred_disp_path = relpath.replace('images/','{}'.format(self.disp_key))
        pred_disp_path = os.path.join(str(relpath).split('images')[0], str(relpath).split('rectified/')[1])
        gt_disp_path = relpath.replace('images/', '{}'.format(self.gt_disp_key))


        # print("left path")
        # print(self.left_image_pathstrings[index])

        # print("subset_root")
        # print(self.subset_root)

        # print("rel path")
        # print(relpath)

        
        return output, relpath, left_image_path, pred_disp_path, gt_disp_path, index
