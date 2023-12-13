import numpy as np

from metaworld.policies.policy import assert_fully_parsed
from metaworld.policies.policy import MoveTo
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_gen_mocap import SawyerPickAndPlaceMultiTaskGenEnvV2

class SawyerPickGenV2Policy():
    def __init__(self, skill='Pick', target_object='milk'):
        self.stage = 0
        self.step_in_stage = 0
        self.target_object = target_object
        
        self.action_list = []
        self.init_target_pos = None
        
        self.action_list.append(MoveTo(self.target_func_above_object, 'gripper_open', error_limit=0.01))
        if target_object == 'bottle':
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_bottle, 'gripper_open', error_limit=0.01, time_limit=100))
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_bottle, 'gripper_close', time_limit=20))
        elif 'red_mug' in target_object:
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_red_mug, 'gripper_open', error_limit=0.01, time_limit=100))
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_red_mug, 'gripper_close', time_limit=30))
        elif 'white_mug' in target_object or 'blue_mug' in target_object:
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_white_mug, 'gripper_open', error_limit=0.01, time_limit=100))
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_white_mug, 'gripper_close', time_limit=30))
        elif target_object == 'milk':
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_milk, 'gripper_open', error_limit=0.01, time_limit=100))
            self.action_list.append(MoveTo(self.target_func_above_target_move_down_milk, 'gripper_close', time_limit=20))
        else:
            self.action_list.append(MoveTo(self.target_func_above_target_move_down, 'gripper_open', error_limit=0.01, time_limit=100))
            self.action_list.append(MoveTo(self.target_func_above_target_move_down, 'gripper_close', time_limit=20))
        self.action_list.append(MoveTo(self.target_func_above_target, 'gripper_close', error_limit=0.01, time_limit=100))

    def _parse_obs(self, obs):
        if isinstance(obs, dict):
            ee_xyz = obs['ee_xyz'][:3]
            gripper_dist = obs['ee_xyz'][3]
            goal = obs['goal']
            target_object_xyz = obs[self.target_object + '_pos']

        else:
            ee_xyz = obs[:3]
            gripper_dist = obs[3]
            goal = obs[-3:]
            if len(obs) <= (4 + 2 * 7 + 3):
                target_object_xyz = obs[4:7]
            else:
                all_objects = SawyerPickAndPlaceMultiTaskGenEnvV2.BLOCKS + SawyerPickAndPlaceMultiTaskGenEnvV2.OBJECTS
                assert self.target_object in all_objects
                target_object_index = all_objects.index(self.target_object)
                start_idx = 4  # ee_xyz + gripper
                obj_pos_index = start_idx + target_object_index * 7
                target_object_xyz = obs[obj_pos_index: obj_pos_index + 3]

        debug = False
        if debug:
            print(f'goal: {np.array_str(goal, precision=3, suppress_small=True)}')
            print(f'ee_xyz: {np.array_str(ee_xyz, precision=3, suppress_small=True)}, gripper: {gripper_dist:.2f}')

        return {
            'ee_xyz': ee_xyz,
            'target_object_xyz': target_object_xyz,
            'goal': goal
        }

    def get_action(self, obs):
        action, end = self.action_list[self.stage].plan_action(self._parse_obs(obs))
        # print(f'stage {self.stage}')
        # Keep track of stages/phases
        if end:
            self.stage += 1
            self.step_in_stage = 0
            print(f'change stage {self.stage}, action list: {len(self.action_list)}')
            
            # if not self.stage == len(self.action_list):
            if self.stage != len(self.action_list):
                self.action_list[self.stage].step_in_stage = 0
            else:
                end = False
                self.stage = len(self.action_list) - 1
                self.action_list[self.stage].step_in_stage = 0
        return action

    def target_func_above_object(self, state):
        if self.init_target_pos is None:
            self.init_target_pos = state['target_object_xyz'].copy()
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.20
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target
    
    def target_func_above_target_move_down(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.03
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57)
            ]
        )
        return target
    
    def target_func_above_target_move_down_bottle(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.07
        angle = (3.14, 0, -1.57)
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57)
            ]
        )
        return target
    
    def target_func_above_target_move_down_red_mug(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.07
        target_xyz[1] += 0.03
        angle = (3.14, 0, -1.57)
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57)
            ]
        )
        return target

    def target_func_above_target_move_down_white_mug(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.03
        target_xyz[1] += 0.03
        angle = (3.14, 0, -1.57)
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57)
            ]
        )
        return target
    
    def target_func_above_target_move_down_milk(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-1] += 0.06
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57)
            ]
        )
        return target

    def target_func_above_target(self, state):
        target_xyz = self.init_target_pos.copy()
        target_xyz[-1] = 0.3
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_place_xy(self, state):
        goal = state['goal'].copy()
        goal[-1] = 0.3
        target = np.hstack(
            [
                goal,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_place_z(self, state):
        goal = state['goal'].copy()
        target = np.hstack(
            [
                goal,
                (3.14, 0, -1.57),
            ]
        )
        return target