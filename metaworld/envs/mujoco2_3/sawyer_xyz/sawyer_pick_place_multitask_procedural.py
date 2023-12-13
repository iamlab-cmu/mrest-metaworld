import numpy as np
import time
from gym.spaces import Box
from typing import Any, List, Mapping, Optional, Tuple
from omegaconf import OmegaConf

import os
from shapely import Polygon
import mujoco
from mujoco import viewer
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco2_3.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

def create_centered_polygon_with_halfsize(size_x: float, size_y: float):
    return np.array([
        [-size_x, -size_y],
        [-size_x, size_y],
        [size_x, size_y],
        [size_x, -size_y],
    ])

class BlockMultiTaskInfo:
    def __init__(self, cfg) -> None:
        self.task_cfg = cfg
        self._all_tasks = self.__class__.all_tasks()

    @staticmethod
    def all_tasks() -> List[str]:
        return  [
            'pick',
            'pick_and_place',
            'push_left', 'push_right', 'push_forward', 'push_backward',
            'reach_left', 'reach_right', 'reach_front', 'reach_back', 'reach_above',
            'pick_and_place_left', 'pick_and_place_right', 'pick_and_place_front', 'pick_and_place_back',
            'push_to_goal',
            'stack',

            'buttonpush',
            'binpick',
        ]

    @property
    def task(self) -> str:
        task = self.task_cfg['task']
        assert task in self._all_tasks, f'Invalid task: {task}'
        return task

    @property
    def target_objects(self) -> str:
        return self.task_cfg['target_object']

    @property
    def stack_target_object(self) -> str:
        return self.task_cfg['stack_on_object']

    @property
    def has_goal_site_for_pick_place(self) -> bool:
        # If we have a goal site then we need to get the color of the goal site
        # and use a different color for the other distractor site
        return False

    @property
    def update_block_colors(self) -> bool:
        return self.task_cfg.get('update_block_colors', False)


class SawyerPickAndPlaceMultiTaskEnvProcV2(SawyerXYZEnv):

    # NOTE: This list should be consistent in how these objects are specified
    # in the XML file.
    # BLOCKS = ['blockA', 'blockB', 'blockC', 'blockD']
    # OBJECTS = ['bread','bottle','coke','pepsi','milk', 'round_mug', 'bottle_sauce']

    BLOCK_COLORS = {
        'block_red': [0.8, 0.0, 0.0, 1],
        'block_blue': [0, 0, 0.8, 1.],
        'block_green': [0, 1, 0, 1.],
        'block_orange': [1, 0.4, 0, 1.],
        'block_yellow': [0.91, 0.84, 0.42, 1.],
        'block_pink': [1.0, 0.57, 0.69, 1.],
        'block_purple': [0.36, 0.25, 0.42, 1.],
        'block_black': [0.15, 0.13, 0.14, 1.],
        # Zero shot colors (never before seen)
        'block_white': [0.9, 0.9, 0.9, 1],
        'block_sky_blue': [0.52, 0.80, 0.92, 1.0],
        'block_olive': [0.50, 0.50, 0.0, 1.0],
        'block_magenta': [0.99, 0.0, 0.99, 1.0],
        'block_brown': [0.58, 0.29, 0.0, 1.],
    }

    def __init__(self, env_cfg, data_collection=False,):

        hand_low = (-0.5, 0.35, 0.07)
        hand_high = (0.5, 1, 0.4)

        self.env_cfg = env_cfg

        self.add_buttonpress_obj = 'task_buttonpress' in env_cfg and env_cfg['task_buttonpress']
        if self.add_buttonpress_obj:
            self.buttonpress_env_cfg = env_cfg['buttonpress_env_cfg']

        self.add_bin_obj = 'task_binpick' in env_cfg and env_cfg['task_binpick']
        if self.add_bin_obj:
            self.binpick_env_cfg = env_cfg['binpick_env_cfg']
            assert not self.add_buttonpress_obj, 'Cannot have both buttonbox and bin'

        # NOTE: Ideally task and envs should be separate.
        self.task_info = BlockMultiTaskInfo(env_cfg)

        mocap_low = np.array([-0.43, 0.35, 0.07])
        mocap_high = np.array([0.43, 0.75, 0.4])

        self.blocks = self.env_cfg['blocks']
        self.objects = self.env_cfg['medium_objects']
        self.only_use_block_objects = env_cfg.get('only_use_block_objects', False)

        if self.only_use_block_objects:
            self._all_objects = self.blocks
        else:
            self._all_objects = self.blocks + self.objects

        self.meta_cfg = OmegaConf.create({
            'goal': {'low': [-0.43, 0.35], 'high': [0.43, 0.75]},
            'gui': False,
            'error_limit': 0.05,
            'collision_thresh': 0.05,
            'reset_steps': 5,
            'episode_len': 300,
        })

        self.block_height = 0.03
        self.object_init_locations = OmegaConf.create({
            'blockA': [-0.4, 0.75, self.block_height],
            'blockB': [0.3, 0.4, self.block_height],
            'blockC': [0.1, 0.4, self.block_height],
            'blockD': [0.2, 0.4, self.block_height],
            'bread': [-0.1, 0.5, self.block_height],
            'bottle': [-0.4, 0.5, 0.065],
            'coke': [-0.2, 0.5, 0.04],
            'pepsi': [0.0, 0.6, 0.04],
            'milk': [0.4, 0.5, 0.06],
            'round_mug': [-0.2, 0.7, 0.02],
            'blue_mug': [-0.2, 0.3, 0.02],
            'red_mug': [-0.3, 0.35, 0.0],
            'white_mug': [-0.3, 0.55, 0.0],
            'blue_mug': [0.3, 0.5, 0.0],
            'cereal': [0.3, 0.6, 0.07],
            'reebok_shoe': [-0.5, 0.7, 0.0],
            'pink_heel': [0.5, 0.3, 0.0],
            'brown_heel': [-0.1, 0., 0.0],
            'green_shoe': [-0.5, 0.3, 0.0],
            'reebok_blue_shoe': [0.0, 0.3, 0.0],
            'reebok_black_shoe': [0.5, 0.7, 0.0],
            'reebok_pink_shoe': [-0.1, 0.3, 0.0],
            'supplement0': [0.1, 0.0, 0.0],
            'supplement1': [0.1, 0.55, 0.0],
            'supplement2': [0.1, 0.7, 0.0],
        })

        total_num_objects = self.number_of_objects()
        super().__init__(
            self._all_objects,
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            total_num_objects=total_num_objects,
        )
        # viewer.launch(self.model, self.data)
        # import ipdb; ipdb.set_trace()
        self.error_limit = self.meta_cfg.error_limit
        self._collision_thresh = self.meta_cfg.collision_thresh

        self._goal_low = np.array(self.meta_cfg.goal.low)
        self._goal_high = np.array(self.meta_cfg.goal.high)

        self.object_sizes_dict = {
            'shoe': create_centered_polygon_with_halfsize(0.35 / 2., 0.15 / 2.),
            'mug': create_centered_polygon_with_halfsize(0.15 / 2., 0.10 / 2.),
        }

        self.hand_init_pos = np.array((0, 0.6, 0.2))
        self.randomize_obj_positions = True

        self.data_collection = data_collection
    
    def number_of_objects(self):
        if self.only_use_block_objects:
            num_blocks = len(self.blocks)
            if self.add_bin_obj:
                return num_blocks + 1
            elif self.add_buttonpress_obj:
                return num_blocks + 1
            else:
                return num_blocks
        else:
            return len(self.blocks) + len(self.objects)
    
    @property
    def random_state(self):
        '''Get env's random state if set or else get global random state.'''
        return self.np_random if hasattr(self, 'np_random') else np.random

    @property
    def task(self):
        '''Return the task for this env.'''
        return self.task_info.task

    def get_all_objects(self):
        return self._all_objects

    def get_target_objects(self):
        return self.task_info.target_objects

    def get_target_object_collision_shape_for_sampling(self, task, object_names):
        objects = []
        obj_th_xy = [0.05, 0.05]
        if task == 'pick':
            if 'shoe' in object_names or 'heel' in object_names:
                obj_polygon = self.object_sizes_dict['shoe']
            elif 'mug' in object_names:
                obj_polygon = self.object_sizes_dict['mug']
            else:
                obj_polygon = create_centered_polygon_with_halfsize(obj_th_xy[0], obj_th_xy[1])
            objects.append(obj_polygon)
        elif task == 'pick_and_place':
            if 'shoe' in object_names or 'heel' in object_names:
                obj_polygon = self.object_sizes_dict['shoe']
            elif 'mug' in object_names:
                obj_polygon = self.object_sizes_dict['mug']
            else:
                obj_polygon = create_centered_polygon_with_halfsize(obj_th_xy[0], obj_th_xy[1])
            goal_polygon = create_centered_polygon_with_halfsize(0.08, 0.08)
            objects.extend([obj_polygon, goal_polygon])
        elif task.startswith('push') or 'reach' in task:
            # NOTE: Specify the magnitude we can move when pushing?
            if 'left' in task or 'right' in task:
                obj_with_clearance_halfsize = (0.2, 0.05)
            elif 'forward' in task or 'backward' in task:
                obj_with_clearance_halfsize = (0.05, 0.2)
            else:
                obj_with_clearance_halfsize = (obj_th_xy[0], obj_th_xy[1])
                # raise ValueError(f'Invalid push direction: {task}')
            obj_polygon = create_centered_polygon_with_halfsize(*obj_with_clearance_halfsize)
            objects.append(obj_polygon)
        elif task == 'stack':
            pick_polygon = create_centered_polygon_with_halfsize(obj_th_xy[0], obj_th_xy[1])
            place_polygon = create_centered_polygon_with_halfsize(0.08, 0.08)
            objects.extend([pick_polygon, place_polygon])
        elif task == 'buttonpush' or task == 'binpick':
            obj_polygon = create_centered_polygon_with_halfsize(obj_th_xy[0], obj_th_xy[1])
            objects.append(obj_polygon)
        elif 'pick_and_place' in task:
            pick_polygon = create_centered_polygon_with_halfsize(obj_th_xy[0], obj_th_xy[1])
            place_polygon = create_centered_polygon_with_halfsize(0.08, 0.08)
            objects.extend([pick_polygon, place_polygon])
        else:
            raise ValueError(f'Invalid task: {task}')

        return objects

    def get_bin_object_sampling_regions(self):
        regions_dict = {
                'left': {
                    'low_xy': (0.1, 0.35),
                    'high_xy': (0.4, 0.75),
                    'bin_x': (-0.3, 0.0)
                },
                'right': {
                    'low_xy': (-0.4, 0.35),
                    'high_xy': (-0.1, 0.75),
                    'bin_x': (0.1, 0.4)
                },
            }
        return regions_dict

    def sample_object_locations(self):

        # sampling_start_time = time.time()

        table_bounds_low_xy = (-0.4, 0.35)
        table_bounds_high_xy = (0.4, 0.75)

        if self.add_buttonpress_obj:
            # table_bounds_low_xy = (-0.0, 0.35)
            table_bounds_low_xy = self.buttonpress_env_cfg['table_bounds_low_xy']

        elif self.add_bin_obj:
            # Since bin object will take up space need to sample blocks in the other ergion
            bin_regions_dict = self.binpick_env_cfg['bin_sampling_region']
            bin_region = self.binpick_env_cfg['bin_region']
            table_bounds_low_xy = [t for t in bin_regions_dict[bin_region]['low_xy']]
            table_bounds_high_xy = [t for t in bin_regions_dict[bin_region]['high_xy']]

        task = self.task
        target_objects = [self.get_target_objects()]
        target_polygons: List[np.ndarray] = self.get_target_object_collision_shape_for_sampling(
            task, target_objects)

        if ('stack' in task) or ('pick_and_place' in task and ('left' in task or 'right' in task or 'back' in task or 'front' in task)):
            target_objects.append(self.task_info.stack_target_object)

        if ('push' in task and ('left' in task or 'right' in task)) or ('pick_and_place' in task and ('left' in task or 'right' in task)):
            goal_bounds_low_xy = (-0.3, 0.35)
            goal_bounds_high_xy = (0.3, 0.75)
        else:
            goal_bounds_low_xy = table_bounds_low_xy
            goal_bounds_high_xy = table_bounds_high_xy

        inserted_object_polygons = []
        object_locations_by_name = dict()
        random_state = self.random_state

        # target_polygons contains the target_objects and some optional goal locations
        for object_idx, object_polygon in enumerate(target_polygons):
            num_tries = 5000
            # Check if what we are placing is an object or a goal location e.g. site
            is_object = object_idx < len(target_objects)
            for try_idx in range(num_tries):
                sample_pos = random_state.uniform(goal_bounds_low_xy, goal_bounds_high_xy)
                sampled_obj_polygon = Polygon(object_polygon + sample_pos)

                valid_obj_polygon = [not o.intersects(sampled_obj_polygon) for o in inserted_object_polygons]
                if all(valid_obj_polygon):
                    inserted_object_polygons.append(sampled_obj_polygon)
                    if is_object:
                        object_locations_by_name[target_objects[object_idx]] = {
                            'position': sample_pos, 'polygon': sampled_obj_polygon}
                    else:
                        object_locations_by_name['goal'] = {
                            'position': sample_pos, 'polygon': sampled_obj_polygon }
                    break

        all_objects = self.get_all_objects()
        for object_name in all_objects:
            # Already placed object
            if object_locations_by_name.get(object_name) is not None:
                continue
            obj_size = self.get_object_size(object_name)
            obj_polygon = create_centered_polygon_with_halfsize(
                obj_size[0] / 2., obj_size[1] / 2.)

            num_tries = 2000
            for try_idx in range(num_tries):
                sample_pos = random_state.uniform(table_bounds_low_xy, table_bounds_high_xy)
                sampled_obj_polygon = Polygon(obj_polygon + sample_pos)
                valid_obj_polygon = [not o.intersects(sampled_obj_polygon) for o in inserted_object_polygons]
                if all(valid_obj_polygon):
                    inserted_object_polygons.append(sampled_obj_polygon)
                    object_locations_by_name[object_name] = {
                        'position': sample_pos, 'polygon': sampled_obj_polygon }
                    break
            
                if try_idx == num_tries - 1:
                    inserted_object_polygons.append(sampled_obj_polygon)
                    object_locations_by_name[object_name] = {
                        'position': sample_pos, 'polygon': sampled_obj_polygon }
            assert object_name in object_locations_by_name, f'Cannot sample position for obj: {object_name}'
            
        obj_locations_by_name = {}
        for object_name, object_location_dict in object_locations_by_name.items():
            if object_name == 'goal':
                # assert task in ('pick_and_place', 'stack'), f'No goal location required for task: {task}'
                goal_pos = object_locations_by_name['goal']['position']
                goal_height = random_state.uniform(0.06, 0.08)
                obj_locations_by_name['site'] = np.array([goal_pos[0], goal_pos[1], goal_height])

            else:
                obj_polygon = object_location_dict['polygon']
                if object_name in self.object_init_locations:
                    obj_height = self.object_init_locations[object_name][2]
                else:
                    obj_height = self.block_height
                obj_locations_by_name[object_name] = np.array(
                    [obj_polygon.centroid.x, obj_polygon.centroid.y, obj_height])

        # Randomly sample site location
        sampling_end_time = time.time()
        # print(f'Time to sample: {sampling_end_time - sampling_start_time:.4f}')
        # pprint.pprint(obj_locations_by_name)

        return obj_locations_by_name

    def get_object_size(self, obj_name: str) -> Tuple[int, int]:
        '''Get object size from object name.'''
        is_block_obj = obj_name.startswith('block')
        if is_block_obj:
            default_obj_size = (0.024, 0.024)
            obj_delta = (0.02, 0.02)
        elif 'bottle' in obj_name or 'bread' in obj_name:
            default_obj_size = (0.04, 0.04)
            obj_delta = (0.02, 0.02)
        elif 'shoe' in obj_name:
            default_obj_size = (0.35, 0.15)
            obj_delta = (0.02, 0.02)
        elif 'mug' in obj_name:
            default_obj_size = (0.15, 0.15)
            obj_delta = (0.02, 0.02)
        else:
            default_obj_size = (0.035, 0.035)
            obj_delta = (0.02, 0.02)
        return (default_obj_size[0] + obj_delta[0], default_obj_size[1] + obj_delta[1])

    def reset_model(self):
        self._reset_hand()
        # viewer.launch(self.model, self.data)
        # import ipdb; ipdb.set_trace()
        # Sample object locations

        if self.randomize_obj_positions:
            self.obj_locations_by_name = self.sample_object_locations()
            self.all_obj_init_pos = np.array([self.obj_locations_by_name[obj_name]
                                              for obj_name in self.get_all_objects()])
        else:
            # self.all_obj_init_pos = [np.array(self.object_init_locations[obj] for obj in self.get_all_objects())]
            self.obj_locations_by_name = self.object_init_locations
            self.all_obj_init_pos = np.array([self.object_init_locations[obj] for obj in self.get_all_objects()])

        # Randomize button object position
        if self.add_buttonpress_obj and self.randomize_obj_positions:
            # buttonbox_x = self.np_random.uniform(-0.4, -0.5)
            buttonbox_x = self.np_random.uniform(*self.buttonpress_env_cfg['buttonbox_x_range'])
            # [0.6, 0.7] is the original training data distribution used (for few demos)
            # buttonbox_y = self.np_random.uniform(0.6, 0.7)
            # This is the harder OOD distribution for eval.
            # buttonbox_y = self.np_random.uniform(0.38, 0.50)
            buttonbox_y = self.np_random.uniform(*self.buttonpress_env_cfg['buttonbox_y_range'])
            buttonbox_z = 0.12
            self.buttonbox_pos = np.array([buttonbox_x, buttonbox_y, buttonbox_z])
            self.sim.model.body_pos[
                self.model.body_name2id('box')] = [buttonbox_x, buttonbox_y, buttonbox_z]

        # Randomize bin object position
        if self.add_bin_obj and self.randomize_obj_positions:
            bin_regions_dict = self.binpick_env_cfg['bin_sampling_region']
            bin_region = self.binpick_env_cfg['bin_region']
            bin_x_range = [t for t in bin_regions_dict[bin_region]['bin_x']]
            bin_x = self.np_random.uniform(*bin_x_range)
            bin_y = self.np_random.uniform(0.4, 0.6)
            bin_z = 0
            self.bin_goal_pos = np.array([bin_x, bin_y, bin_z])
            self.sim.model.body_pos[
                self.model.body_name2id('bin_goal')] = [bin_x, bin_y, bin_z]

        self._set_obj_xyz(self.all_obj_init_pos)

        self._goal_site_pos = self._set_goal_site(self.task_info.task)
        self._target_pos = self._set_goal()

        if self.task_info.update_block_colors:
            for block_id in ['A', 'B', 'C', 'D']:
                if f'block{block_id}_config' not in self.env_cfg:
                    continue
                block_color_name = self.env_cfg[f'block{block_id}_config']['color']
                block_color = SawyerPickAndPlaceMultiTaskEnvProcV2.BLOCK_COLORS[block_color_name]
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'block{block_id}_g0_vis')
                self.model.geom_rgba[geom_id] = np.array(block_color)

        return self._get_obs()
    
    def _set_goal_site(self, task: str):
        goal_pos = None
        place_site_name = 'placeSiteA'
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, place_site_name)
        if 'site' in self.obj_locations_by_name:
            goal_pos = self.obj_locations_by_name['site']
            self._set_pos_site(place_site_name, goal_pos)
            self.model.site_pos[site_id] = goal_pos
        else:
            # Make the site placeholder invisible
            rgba = np.copy(self.model.site_rgba[site_id])
            rgba[-1] = 0.0
            self.model.site_rgba[site_id] = rgba

        return goal_pos

    def _set_obj_xyz(self, pos):
        assert len(pos) == len(self.get_all_objects()), 'Invalid number of objects specified.'
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # Why 9?
        qpos_start_index = 9
        qvel_start_index = 9
        qpos_size, qvel_size = 3, 6
        if self.add_buttonpress_obj:
            assert len(qpos) == qpos_start_index + 7 * len(pos) + 1
            assert len(qvel) == qvel_start_index + qvel_size * len(pos) + 1
            qpos[-1] = 0
            qvel[-1] = 0
        else:
            assert len(qpos) == qpos_start_index + 7 * len(pos)
            assert len(qvel) == qvel_start_index + qvel_size * len(pos)
        for p in pos:
            # qpos[9:12] = pos.copy()
            # qvel[9:15] = 0
            qpos[qpos_start_index:qpos_start_index + qpos_size] = p.copy()
            qvel[qvel_start_index:qvel_start_index + qvel_size] = 0
            qpos_start_index += 7
            qvel_start_index += qvel_size

        self.set_state(qpos, qvel)


    def _set_goal(self):
        task_lower = self.task.lower()
        goal = np.zeros((3,))

        # TODO(Mohit): Should we have a list of target objects?
        # target_object = self.get_target_objects()[0]
        target_object = self.get_target_objects()
        stack_on_object_name = self.env_cfg.get('stack_on_object', '')
        stack_on_object_name = f'{stack_on_object_name}_name'

        if 'buttonpush' in task_lower:
            goal = self.buttonbox_pos + np.array([0.12, 0., -0.05])
            self._buttonbox_target_pos = self._get_site_pos('hole')
        
        elif 'binpick' in task_lower:
            # goal = self.bin_goal_pos + np.array([0., 0., 0.02])
            goal = self.get_body_com('bin_goal') + np.array([0., 0., 0.06])

        elif 'pick' in task_lower and 'place' in task_lower and 'side' in task_lower:
            raise NotImplementedError('Task not implemented')
        
        elif 'pick' in task_lower and 'place' in task_lower and 'left' in task_lower:
            goal = self.get_body_com(stack_on_object_name).copy()
            goal[0] += 0.08
            goal[-1] = self.object_init_locations[self.env_cfg['stack_on_object']][2]+0.03
        elif 'pick' in task_lower and 'place' in task_lower and 'right' in task_lower:
            goal = self.get_body_com(stack_on_object_name).copy()
            goal[0] -= 0.08
            goal[-1] = self.object_init_locations[self.env_cfg['stack_on_object']][2]+0.03
        elif 'pick' in task_lower and 'place' in task_lower and 'front' in task_lower:
            goal = self.get_body_com(stack_on_object_name).copy()
            goal[1] -= 0.08
            goal[-1] = self.object_init_locations[self.env_cfg['stack_on_object']][2]+0.03
        elif 'pick' in task_lower and 'place' in task_lower and 'back' in task_lower:
            goal = self.get_body_com(stack_on_object_name).copy()
            goal[1] += 0.08
            goal[-1] = self.object_init_locations[self.env_cfg['stack_on_object']][2]+0.03
        
        elif 'pick' in task_lower and 'place' in task_lower:
            goal = self._get_site_pos('placeSiteA')
        elif 'pick' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            if target_object != 'bottle' and target_object != 'milk' and target_object != 'bread':
                goal[-1] = 0.3 - 0.03
            elif target_object == 'bread':
                goal[-1] = 0.3 - 0.03
            elif target_object == 'bottle' or target_object == 'round_mug' or target_object == 'bottle_sauce':
                goal[-1] = 0.3 - 0.07
            elif target_object == 'milk':
                goal[-1] = 0.3 - 0.06
        elif 'push' in task_lower and 'goal' in task_lower:
            raise NotImplementedError('Task not implemented')
        elif 'push' in task_lower and 'forward' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[1] += 0.1
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'push' in task_lower and 'backward' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[1] -= 0.1
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'push' in task_lower and 'left' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[0] -= 0.1
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'push' in task_lower and 'right' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[0] += 0.1
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'reach' in task_lower and 'left' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[0] += 0.06
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'reach' in task_lower and 'right' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[0] -= 0.06
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'reach' in task_lower and 'front' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[1] -= 0.06
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'reach' in task_lower and 'back' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            goal[1] += 0.06
            goal[-1] = self.object_init_locations[target_object][2]
        elif 'reach' in task_lower and 'above' in task_lower:
            goal = self.get_body_com(f'{target_object}_main').copy()
            if target_object != 'bottle' and target_object != 'milk':
                goal[-1] += 0.05
            elif target_object == 'bottle' or target_object == 'round_mug' or target_object == 'bottle_sauce':
                goal[-1] += 0.09
            elif target_object == 'milk':
                goal[-1] += 0.08

        elif 'stack' in task_lower:
            pos = self.get_body_com(stack_on_object_name).copy()
            goal[:2] = pos[:2]
            # goal[-1] = pos[-1] + 0.12
            goal[-1] = pos[-1] + 0.09
        else:
            raise NotImplementedError('Task not implemented')

        self.goal = goal.copy()
        return goal

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        # TODO: Add mask for objects that are not present in the scene.
        objects = np.hstack([self.get_body_com(f'{obj_name}_main')
                             for obj_name in self.get_all_objects()])
        # import ipdb; ipdb.set_trace()
        if self.add_buttonpress_obj:
            # self.get_body_com('button') + np.array([.0, -.193, .0])
            objects = np.r_[objects, self.get_body_com('button') + np.array([.193, 0., .0])]

        elif self.add_bin_obj:
            objects = np.r_[objects, self.get_body_com('bin_goal')]

        return objects


    def _get_quat_objects(self):
        objects = np.hstack([self.get_body_quat(obj_name)
                             for obj_name in self.get_all_objects()])
        if self.add_buttonpress_obj:
            objects = np.r_[objects,
                self.get_body_quat('button')]
        
        elif self.add_bin_obj:
            objects = np.r_[objects, 
                self.get_body_quat('bin_goal')]

        return objects

    def evaluate_state(self, obs, action):
        (
            reward,
            near_object,
            obj_to_target,
            gripper_open_without_block
        ) = self.compute_reward(action, obs)

        dist_to_goal_metric = (obj_to_target <= self.error_limit)
        gripper_vel = np.linalg.norm(self.data.cvel[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')])
        gripper_vel_metric = gripper_vel < 0.005
        

        if self.data_collection:
            if 'stack' in self.task:
                success = float( dist_to_goal_metric and gripper_open_without_block)
            elif self.task.startswith('push'):
                success = float((obj_to_target <= self.error_limit+0.03) and gripper_vel_metric)
            elif 'reach' in self.task:
                ee_to_goal = np.linalg.norm(obs['ee_xyz'][:3] - self._target_pos) <= self.error_limit
                success = float(ee_to_goal and gripper_vel_metric)
            elif 'pick' in self.task and 'place' in self.task:
                success = float( (obj_to_target <= self.error_limit+0.01) and gripper_open_without_block)
            elif self.task == 'buttonpush':
                button_pos = obs[-(3 + 7):-(3 + 4)]
                obj_to_target = abs(self._buttonbox_target_pos[0] - button_pos[0])
                success = obj_to_target <= 0.0095
            
            elif self.task == 'binpick':
                success = float( dist_to_goal_metric and gripper_open_without_block)

            else:
                success = float(obj_to_target <= self.error_limit+0.01)
        else:
            if 'stack' in self.task:
                success = float( dist_to_goal_metric)
            elif self.task.startswith('push'):
                success = float(dist_to_goal_metric)
                success = float((obj_to_target <= self.error_limit+0.03))
            elif 'reach' in self.task:
                ee_to_goal = np.linalg.norm(obs['ee_xyz'][:3] - self._target_pos) <= self.error_limit
                success = float(ee_to_goal)
            elif 'pick' in self.task and 'place' in self.task:
                success = float( dist_to_goal_metric )

            elif self.task == 'buttonpush':
                button_pos = obs[-(3 + 7):-(3 + 4)]
                obj_to_target = abs(self._buttonbox_target_pos[0] - button_pos[0])
                success = obj_to_target <= 0.0095

            elif self.task == 'binpick':
                success = float( dist_to_goal_metric)

            else:
                success = float(dist_to_goal_metric)

        info = {
            'success': success,
            'near_object': float(near_object),
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    def _parse_target_object_position_from_obs(self, obs):
        '''Parse target object position from observation.'''
        target_obj = self.get_target_objects()
        target_obj_index = self._all_objects.index(target_obj)
        start_idx = 4
        obj_pos_idx = start_idx + target_obj_index * 7
        return obs[obj_pos_idx: obj_pos_idx + 3]

    def compute_reward(self, action, obs):
        hand = obs['ee_xyz'][:3]

        if self.add_buttonpress_obj and self.task == 'buttonpush':
            target_to_obj = np.linalg.norm(hand - self.goal)

            # NOTE: 4cm is the default used
            near_object = np.linalg.norm(self.goal - hand) < 0.04
        
        elif self.add_bin_obj and self.task == 'binpick':
            # obj = self._parse_target_object_position_from_obs(obs)
            target_obj = self.get_target_objects()
            obj = obs[f'{target_obj}_pos']
            target_to_obj = np.linalg.norm(obj - self._target_pos)
            # NOTE: 4cm is the default used
            near_object = np.linalg.norm(obj - hand) < 0.04

        else:
            # obj = self._parse_target_object_position_from_obs(obs)
            target_obj = self.get_target_objects()
            target_to_obj = np.linalg.norm(obs[f'{target_obj}_pos'] - self._target_pos)

            # NOTE: 4cm is the default used
            near_object = np.linalg.norm(obs[f'{target_obj}_pos'] - hand) < 0.04

        gripper_open_without_block = (obs['ee_xyz'][3]>0.7)

        reward = 0. # TODO: Implement reward
        return reward, near_object, target_to_obj, gripper_open_without_block

    def _get_obs(self):
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the
            goal position to form a single flat observation.

        Returns:
            np.ndarray: The flat observation array (39 elements)
        """
        pos_hand = self.get_endeff_pos()

        # Reset goal location for the first 5 steps (since setting it directly at reset)
        # doens't update it correctly.
        if self.curr_path_length < 5:
            self._goal_site_pos = self._set_goal_site(self.task_info.task)
            self._target_pos = self._set_goal()

        finger_right, finger_left = (
            self._get_site_pos('robot0_rightEndEffector'),
            self._get_site_pos('robot0_leftEndEffector')
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
        # do frame stacking
        obs_dict = {}
        obs_dict['ee_xyz'] = np.hstack([pos_hand, gripper_distance_apart])
        obs_dict['goal'] = np.copy(self._get_pos_goal())

        for obj_name in self.get_all_objects():
            obj_pos = self.get_body_com(f'{obj_name}_main').copy()
            obj_quat = self.get_body_quat(f'{obj_name}_main').copy()
            obs_dict[f'{obj_name}_pos'] = np.copy(obj_pos)
            obs_dict[f'{obj_name}_quat'] = np.copy(obj_quat)
        
        # viewer.launch(self.model, self.data)
        # print('cereal pos:', obs_dict[f'cereal_pos'])
        # print('shoe pos:', obs_dict[f'reebok_shoe_pos'])
        # print('mug pos:', obs_dict[f'mug_pos'])
        # print('block pos:', obs_dict[f'blockA_pos'])
        return obs_dict