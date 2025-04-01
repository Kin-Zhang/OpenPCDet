"""
# This file is adapted from OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from typing import List, Callable
from functools import partial
from tools.h5sf import H5Dataset
import argparse
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pathlib import Path
from tools.visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance

VIEW_FILE = f"av2.json"

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

color_map_hex = ['#a6cee3', '#de2d26', '#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',\
                 '#cab2d6','#6a3d9a','#ffff99','#b15928', '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3',\
                 '#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

color_map = [hex_to_rgb(color) for color in color_map_hex]

class MyVisualizer:
    def __init__(self, view_file=None, window_title="Default", save_folder="logs/imgs"):
        self.params = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_title)
        self.view_file = view_file

        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.save_img_folder = save_folder
        os.makedirs(self.save_img_folder, exist_ok=True)
        print(
            f"\n{window_title.capitalize()} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t[ESC/Q] to exit\n"
            "\t    [P] to save screen and viewpoint\n"
            "\t    [N] to step\n"
        )
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback(["P"], self._save_screen)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)

    def show(self, assets: List):
        self.vis.clear_geometries()

        for asset in assets:
            self.vis.add_geometry(asset)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())

        self.vis.update_renderer()
        self.vis.poll_events()
        self.vis.run()
        self.vis.destroy_window()

    def update(self, assets: List, clear: bool = True):
        if clear:
            self.vis.clear_geometries()

        for asset in assets:
            self.vis.add_geometry(asset, reset_bounding_box=False)
            self.vis.update_geometry(asset)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())
            self.reset_bounding_box = False

        self.vis.update_renderer()
        while self.block_vis:
            self.vis.poll_events()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _quit(self, vis):
        print("Destroying Visualizer. Thanks for using ^v^.")
        vis.destroy_window()
        os._exit(0)

    def _save_screen(self, vis):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        png_file = f"{self.save_img_folder}/ScreenShot_{timestamp}.png"
        view_json_file = f"{self.save_img_folder}/ScreenView_{timestamp}.json"
        with open(view_json_file, 'w') as f:
            f.write(vis.get_view_status())
        vis.capture_screen_image(png_file)
        print(f"ScreenShot saved to: {png_file}, Please check it.")


class MyMultiVisualizer(MyVisualizer):
    def __init__(self, view_file=None, flow_mode=['flow'], screen_width=2500, screen_height = 1375):
        self.params = None
        self.view_file = view_file
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.playback_direction = 1 # 1:forward, -1:backward

        self.vis = []
        # self.o3d_vctrl = []

        # Define width and height for each window
        window_width = screen_width // 2
        window_height = screen_height // 2
        # Define positions for the four windows
        epsilon = 150
        positions = [
            (0, 0),  # Top-left
            (screen_width - window_width + epsilon, 0),  # Top-right
            (0, screen_height - window_height + epsilon),  # Bottom-left
            (screen_width - window_width + epsilon, screen_height - window_height + epsilon)  # Bottom-right
        ]

        for i, mode in enumerate(flow_mode):
            window_title = f"view {'ground truth flow' if mode == 'flow' else f'{mode} flow'}, `SPACE` start/stop"
            v = o3d.visualization.VisualizerWithKeyCallback()
            v.create_window(window_name=window_title, width=window_width, height=window_height, left=positions[i%len(positions)][0], top=positions[i%len(positions)][1])
            # self.o3d_vctrl.append(ViewControl(v.get_view_control(), view_file=view_file))
            self.vis.append(v)

        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["D"], self._next_frame)
        self._register_key_callback(["A"], self._prev_frame)
        print(
            f"\n{window_title.capitalize()} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t[ESC/Q] to exit\n"
            "\t    [P] to save screen and viewpoint\n"
            "\t    [D] to step next\n"
            "\t    [A] to step previous\n"
        )

    def update(self, assets_list: List, clear: bool = True):
        if clear:
            [v.clear_geometries() for v in self.vis]

        for i, assets in enumerate(assets_list):
            [self.vis[i].add_geometry(asset, reset_bounding_box=False) for asset in assets]
            self.vis[i].update_geometry(assets[-1])

        if self.reset_bounding_box:
            [v.reset_view_point(True) for v in self.vis]
            if self.view_file is not None:
                # [o.read_viewTfile(self.view_file) for o in self.o3d_vctrl]
                [v.set_view_status(open(self.view_file).read()) for v in self.vis]
            self.reset_bounding_box = False

        [v.update_renderer() for v in self.vis]
        while self.block_vis:
            [v.poll_events() for v in self.vis]
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            [v.register_key_callback(ord(str(key)), partial(callback)) for v in self.vis]
    def _next_frame(self, vis):
        self.block_vis = not self.block_vis
        self.playback_direction = 1
    def _prev_frame(self, vis):
        self.block_vis = not self.block_vis
        self.playback_direction = -1


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument("--flow_mode", type=list, default=['raw', 'flow'])
    parser.add_argument("--tone", type=str, default='bright')
    parser.add_argument("--point_size", type=float, default=3.0)
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    assert isinstance(args.flow_mode, list), "this script needs a list as flow_mode"
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    dataset = H5Dataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), flow_mode=args.flow_mode, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(dataset)}')

    o3d_vis = MyMultiVisualizer(view_file=VIEW_FILE, flow_mode=args.flow_mode)

    for v in o3d_vis.vis:
        opt = v.get_render_option()
        if args.tone == 'bright':
            # background_color = np.asarray([216, 216, 216]) / 255.0  # offwhite
            background_color = np.asarray([1, 1, 1])
            pcd_color = [0.25, 0.25, 0.25]
        elif args.tone == 'dark':
            background_color = np.asarray([80/255, 90/255, 110/255])  # dark
            pcd_color = [1., 1., 1.]

        opt.background_color = background_color
        opt.point_size = args.point_size

    data_id = args.start_id
    pbar = tqdm(range(0, len(dataset)))
    while data_id >= 0 and data_id < len(dataset):
        pcd_list = []
        for mode in args.flow_mode:
            # data = dataset[data_id]
            data = dataset.__getitem__(data_id, current_flow_mode=mode)
            pcd = o3d.geometry.PointCloud()
            for vis_id in np.unique(data['lidar_id']):
                mask = vis_id == data['lidar_id']
                single_pcd = o3d.geometry.PointCloud()
                single_pcd.points = o3d.utility.Vector3dVector(data['points'][mask, :3])
                single_pcd.paint_uniform_color(color_map[(vis_id) % len(color_map)])
                pcd += single_pcd
            # pcd.points = o3d.utility.Vector3dVector(data['points'][:, :3])
            # pcd.colors = o3d.utility.Vector3dVector(np.array([hex_to_rgb(color_map_hex[id]) for id in data['lidar_id']]))
            # pcd.paint_uniform_color([1.0, 1.0, 1.0])
            
            line_set_list = []
            for i in range(data['pred_boxes'].shape[0]):
                if data['pred_scores'][i] > 0.3:
                    line_set, _ = translate_boxes_to_open3d_instance(data['pred_boxes'][i])
                    line_set.paint_uniform_color((1, 0, 0))
                    line_set_list.append(line_set)
            pcd_list.append([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)]+line_set_list)
            
        o3d_vis.update(pcd_list)

        # now_scene_id = data['scene_id']
        # pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        pbar.set_description(f"id: {data_id}")

        data_id += o3d_vis.playback_direction
        pbar.update(o3d_vis.playback_direction)

            
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
