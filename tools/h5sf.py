import argparse
import glob
from pathlib import Path

# try:
import open3d
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch, h5py, os, sys, pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class H5Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None, flow_mode='raw'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.flow_mode = flow_mode
        self.rootdir = root_path
        with open(os.path.join(self.rootdir, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.data_index)

    def egopts_mask(self, pts, min_bound=[-9.5, -3/2, 0], max_bound=[5, 2.760004/2, 5]):
        """
        Input:
            pts: (N, 3)
        Output:
            mask: (N, ) indicate the points that are outside the egopts
        """
        mask = ((pts[:, 0] > min_bound[0]) & (pts[:, 0] < max_bound[0])
                & (pts[:, 1] > min_bound[1]) & (pts[:, 1] < max_bound[1])
                & (pts[:, 2] > min_bound[2]) & (pts[:, 2] < max_bound[2]))
        return ~mask
    
    def __getitem__(self, index, current_flow_mode=None):

        scene_id, timestamp = self.data_index[index]
        if index+1 < len(self.data_index):
            scene_id1, timestamp1 = self.data_index[index+1]
            if scene_id != scene_id1:
                index -= 1
                return self.__getitem__(index, current_flow_mode=current_flow_mode)
        with h5py.File(os.path.join(self.rootdir, f'{scene_id}.h5'), 'r') as f:
            key = str(timestamp)
            pc = f[key]['lidar'][:]

            delta_t = f[key]['lidar_dt'][:]

            # for scania only ----->
            ego_mask = ~self.egopts_mask(pc)
            gm = f[key]['ground_mask'][:]
            ego_mask = ego_mask | gm
            pc = pc[~ego_mask] 
            
            # for scania only ----->
            pc[:, 2] -= 1.73 # only need for shift Scania data to the same level as KITTI
            delta_t = delta_t[~ego_mask]
            
            # others.
            # ego_maske = np.zeros((pc.shape[0], 1), dtype=np.bool_) # if not scania we don't need remove ground.

            if current_flow_mode is not None:
                flow_mode = current_flow_mode
            else:
                flow_mode = self.flow_mode
            if flow_mode in f[key]:
                # print('Using flow mode:', flow_mode)
                pose0 = f[key]['pose'][:]
                pose1 = f[str(timestamp1)]['pose'][:]
                ego_pose = np.linalg.inv(pose1) @ pose0
                pose_flow = pc[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc[:, :3]
                dt0_raw = f[key]['lidar_dt'][:]

                flow = f[key][flow_mode][:][~ego_mask] - pose_flow
                dt0 = max(dt0_raw) - dt0_raw # we want to see the last frame. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
                ref_pc = pc[:,:3] + (flow/0.1) * dt0[:, None][~ego_mask]
            else:
                ref_pc = pc[:,:3]

            pred_boxes = np.zeros((0, 7), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int32)
            if 'bbox_'+flow_mode in f[key]:
                pred_boxes = f[key]['bbox_'+flow_mode][:]
            if 'bbox_score_'+flow_mode in f[key]:
                if np.isscalar(f[key]['bbox_score_'+flow_mode][()]):
                    pred_scores = np.array([f[key]['bbox_score_'+flow_mode][()]])
                else:
                    pred_scores = f[key]['bbox_score_'+flow_mode][:]
            if 'bbox_labels_'+flow_mode in f[key]:
                pred_labels = f[key]['bbox_labels_'+flow_mode][:]
        pc[:, :3] = ref_pc
        if pc.shape[1] == 4:
            # normalize intensity
            # pc[:, 3] = pc[:, 3] / max(pc[:, 3])
            pc[:, 3] = 0
        else:
            print("No intensity channel found! It may be really worse result here.")
            pc = np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])

        # Ajinkya: nuscenes expects delta_t: comment this out for kitti etc.
        pc = np.hstack([pc, delta_t[:, None]])

        input_dict = {
                'points': pc,
                # 'scene_id': scene_id,
                # 'timestamp': timestamp,
                'frame_id': index,
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
            }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def save_bbox(self, index, pred_boxes, pred_scores=None, pred_labels=None):
        scene_id, timestamp = self.data_index[index]
        
        with h5py.File(os.path.join(self.rootdir, f'{scene_id}.h5'), 'r+') as f:
            key = str(timestamp)
            if 'bbox_'+self.flow_mode in f[key]:
                del f[key]['bbox_'+self.flow_mode]
            if 'bbox_score_'+self.flow_mode in f[key]:
                del f[key]['bbox_score_'+self.flow_mode]
            if 'bbox_labels_'+self.flow_mode in f[key]:
                del f[key]['bbox_labels_'+self.flow_mode]
            f[key].create_dataset('bbox_'+self.flow_mode, data=np.array(pred_boxes).astype(np.float32))
            f[key].create_dataset('bbox_score_'+self.flow_mode, data=np.array(pred_scores).astype(np.float32))
            f[key].create_dataset('bbox_labels_'+self.flow_mode, data=np.array(pred_labels).astype(np.int32))
            
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument("--flow_mode", type=str, default='raw')
    parser.add_argument("--vis", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=True)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = H5Dataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), flow_mode=args.flow_mode, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # for idx, data_dict in enumerate(demo_dataset):
        for idx in range(args.start_id, len(demo_dataset)):
            logger.info(f'Visualized data_id: \t{idx}, scene_id: {demo_dataset.data_index[idx][0]}, timestamp: {demo_dataset.data_index[idx][1]}')
            data_dict = demo_dataset[idx]
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if args.vis:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
            
            if args.save:
                # save the bbx
                demo_dataset.save_bbox(idx, pred_boxes=pred_dicts[0]['pred_boxes'].cpu().numpy(), pred_scores=pred_dicts[0]['pred_scores'].cpu().numpy(), pred_labels=pred_dicts[0]['pred_labels'].cpu().numpy())
            
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
