import os
from omegaconf import OmegaConf
import numpy as np
import random
from moviepy.editor import ImageSequenceClip

try:
    import cv2
except:
    print('OpenCV not found')

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_lock_v2 import SawyerDoorLockEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_v2 import SawyerDoorEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2. sawyer_window_open_v2_ss_gen_env import SawyerWindowOpenGenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_ss_gen_env_v2 import SawyerDrawerOpenGenEnvV2
from r3meval.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_multiobj_multitask_env_gen_mocap import SawyerMultiObjectMultiTaskGenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_multiobj_multitask_multishape_env_gen import SawyerMultiObjectMultiTaskMultiShapeGenEnvV2
from metaworld.policies import *

if __name__ == "__main__":
    
    # env = SawyerDoorLockEnvV2()
    # env = SawyerWindowOpenGenEnvV2()
    # np.random.seed(113)
    # random.seed(113)
    env_config = {

        # 'skill': 'pick_and_place',
        # 'task': 'pick_and_place',
        # "target_object": 'blockB',

        # 'skill': 'pick',
        # 'task': 'pick',
        # "target_object": 'blockC',

        # 'skill': 'stick_door_close',
        # 'task': 'stick_door_close',
        # "target_object": 'door_small',
        # "auxiliary_objects": 'stickB',

        # 'skill': 'push_left',
        # 'task': 'push_left',
        # "target_object": 'blockA',

        # 'skill': 'stack',
        # 'task': 'stack',
        # "target_object": 'blockA',
        # "stack_on_object": 'blockC',

        # 'skill': 'window_close',
        # 'task': 'window_close',
        # "target_object": 'window',

        # 'skill': 'window_open',
        # 'task': 'window_open',
        # "target_object": 'window',
        
        # 'skill': 'door_lock_goal',
        # 'task': 'door_lock_goal',
        # "target_object": 'door',

        # 'skill': 'door_open_goal',
        # 'task': 'door_open_goal',
        # "target_object": 'door_small',

        # 'skill': 'door_close_goal',
        # 'task': 'door_close_goal',
        # "target_object": 'door',

        # 'skill': 'drawer_open_goal',
        # 'task': 'drawer_open_goal',
        # "target_object": 'drawer',

        'skill': 'drawer_close_goal',
        'task': 'drawer_close_goal',
        "target_object": 'drawer',

        # 'skill': 'put_in_drawer',
        # 'task': 'put_in_drawer',
        # "target_object": 'blockA',
        # "stack_on_object": 'drawer_small',

        # 'skill': 'put_in_open_drawer',
        # 'task': 'put_in_open_drawer',
        # "target_object": 'blockA',
        # "auxiliary_objects": 'drawer',

        # 'skill': 'push_in_open_drawer',
        # 'task': 'push_in_open_drawer',
        # "target_object": 'blockA',
        # "auxiliary_objects": 'drawer',

        # 'skill': 'nut_pick',
        # 'task': 'nut_pick',
        # "target_object": 'RoundNut',

        # 'skill': 'put_nut_in_door',
        # 'task': 'put_nut_in_door',
        # "target_object": 'RoundNut',
        # "auxiliary_objects": 'door',

        # 'skill': 'faucet_rotate_goal',
        # 'task': 'faucet_rotate_goal',
        # "target_object": 'faucetBase',

        # 'skill': 'peg_insert',
        # 'task': 'peg_insert',
        # "target_object": 'peg',

        'only_use_block_objects': True,
        'update_block_colors': False,

        'big_objects': ['drawer'],
        'medium_objects': [],
        'sticks': [],
        'small_objects': [],
        # 'small_objects': [],

        'blocks': ['blockA', 'blockB'],
    }

    # env = SawyerMultiObjectMultiTaskGenEnvV2(env_config, data_collection=True)
    env = SawyerMultiObjectMultiTaskMultiShapeGenEnvV2(env_config, data_collection=True)
    
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    save_gif = True
    use_opencv_to_render = False

    # policy = SawyerDoorLockV2Policy()
    # policy = SawyerDoorOpenV2Policy()
    # policy = SawyerDoorLockV2GenPolicy()
    # policy = SawyerDrawerOpenV2GenPolicy()
    # policy = SawyerWindowOpenGenV2Policy()
    # policy = SawyerWindowOpenV2Policy()
    n_trajs = 1
    total_succ = 0
    for i in range(n_trajs):
        policy = SawyerMultitaskV2Policy(env_config['skill'], env_config['target_object'])

        step = 0
        H, W = 256, 256
        cam_name = 'eye_in_hand'

        env.reset_model()
        obs = env.reset()
        
        img = env.sim.render(height=H,width=W,camera_name=cam_name)
        img = img[::-1, :, :]
        success = False
        imgs = [img]
        while (not success) and (step < 250):
        # while (step < 200):
            # a = np.random.uniform(low=0.0, high=10.0, size=4)
            a = policy.get_action(obs)
            # print("window:", obs['window_pos'])
            next_obs, r, done, info = env.step(a)
            # img = env.get_image()
            img = env.sim.render(height=H, width=W, camera_name=cam_name)
            img = img[::-1, :, :]

            if use_opencv_to_render:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow(f'{type(env).__name__}', img)
                cv2.waitKey(1)

            imgs.append(img)
            obs = next_obs.copy()
            step += 1
            
            if info['success']:
                success = info['success']
                success = True
                print(f"================Traj {i}: SUCCESS at step: {step}===============")
                total_succ += 1

        if save_gif:
            print("saving gif")
            skill=env_config['skill']
            tg=env_config['target_object']
            filename = f'./media/sawyer_{skill}_gen_{i}_{success}_{tg}.gif'
            cl = ImageSequenceClip(imgs[:200], fps=20)
            cl.write_gif(filename, fps=20)

    print(f"Total success: {total_succ}/{n_trajs}")
    if use_opencv_to_render:
        cv2.destroyAllWindows()
