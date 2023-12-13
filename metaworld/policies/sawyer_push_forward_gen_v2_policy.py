import numpy as np

from metaworld.policies.policy import MoveTo
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_gen_mocap import SawyerPickAndPlaceMultiTaskGenEnvV2

class SawyerPushForwardGenV2Policy():
    def __init__(self, skill='Push forward', target_object='milk'):
        self.stage = 0
        self.step_in_stage = 0
        self.target_object = target_object
        
        self.action_list = []
        self.init_target_pos = None

        self.action_list.append(MoveTo(self.target_func_move_behind, 'gripper_close', error_limit=0.01))
        self.action_list.append(MoveTo(self.target_func_move_behind_down, 'gripper_close', error_limit=0.01))
        self.action_list.append(MoveTo(self.target_func_push, 'gripper_close', time_limit=100, p=7.))

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

    def target_func_move_behind(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-2] -= 0.06
        target_xyz[-1] += 0.20
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_move_behind_down(self, state):
        target_xyz = state['target_object_xyz'].copy()
        target_xyz[-2] -= 0.06
        target_xyz[-1] += 0.06
        # target_xyz[-1] += 0.20
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target

    def target_func_push(self, state):
        target_xyz = state['goal'].copy()
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, -1.57),
            ]
        )
        return target