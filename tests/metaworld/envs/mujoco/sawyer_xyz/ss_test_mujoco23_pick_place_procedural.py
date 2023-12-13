import numpy as np
from moviepy.editor import ImageSequenceClip

try:
    import cv2
except:
    print('OpenCV not found')

from metaworld.envs.mujoco2_3.sawyer_xyz.sawyer_pick_place_multitask_procedural import SawyerPickAndPlaceMultiTaskEnvProcV2
from r3meval.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy
from metaworld.policies import SawyerMultitaskV2Policy


if __name__ == "__main__":
    
    # env_config = {
    #     "target_object": "reebok_blue_shoe", 
    #     "stack_on_object": "blockA", 
    #     "skill": "reach", 
    #     "task": "reach",
    #     "only_use_block_objects": False,
    #     "blocks": ['blockA'],
    #     "medium_objects": ['reebok_black_shoe', 'reebok_blue_shoe', 'green_shoe', 'pink_heel']}
    env_config = {
        "auxiliary_objects": '',
        "big_objects": [],
        "blockA_config": {"color": "block_red"},
        "blockB_config": {"color": "block_blue"},
        "blocks": ['blockA', 'blockB'],
        "camera_names": ['left_cap2', 'robot0_eye_in_hand_90'],
        "distractor_cfg_key": 1,
        "has_small_objects": False,
        "medium_objects": ["green_shoe","reebok_blue_shoe","red_mug","white_mug"],
        "multiobj": True,
        "num_demos_per_env": 4,
        "procedural": True,
        "randomize_medium_object_colors": True,
        "skill": "reach_above",
        "small_objects": [],
        "sticks": [],
        "target_object": "green_shoe",
        "task": "reach_above",
        "task_command_color": "Pick and place red object",
        "task_command_type": "Pick and place red block",
        "update_block_colors": True,
        "update_stick_colors": True,
    }
    env = SawyerPickAndPlaceMultiTaskEnvProcV2(env_config, data_collection=True)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    use_opencv_to_render = False
    save_gif = True

    n_trajs = 1
    total_succ = 0
    H, W = 256, 256
    # cam_name = 'robot0_eye_in_hand_90'
    cam_name = 'left_cap2'
    for i in range(n_trajs):
    
        env = SawyerPickAndPlaceMultiTaskEnvProcV2(env_config, data_collection=True)
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        step = 0
        
        policy = SawyerMultitaskV2Policy(skill=env_config['skill'], target_object=env_config['target_object'])

        env.reset_model()
        obs = env.reset()
        img = env.render(offscreen=True, camera_name=cam_name, resolution=(H, W))
        # img = img[::-1, :, :]
        success = False
        imgs = [img]
        
        while (not success) and (step < 200):
            # a = np.random.uniform(low=0.0, high=10.0, size=4)
            a = policy.get_action(obs)
            # a = np.random.normal(a, 0.07 * 2)
            next_obs, r, done, info = env.step(a)

            # from mujoco import viewer
            # viewer.launch(env.model, env.data)
            # img = env.get_image()
            img = env.render(offscreen=True, camera_name=cam_name, resolution=(H, W))
            # img = img[::-1, :, :]

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
                print(f"================SUCCESS at step: {step}===============")
                total_succ += 1

        if save_gif:
            print("saving gif")
            skill=env_config['skill']
            tg=env_config['target_object']
            filename = f'./media/sawyer_{skill}_gen_{i}_{success}_{tg}.gif'
            cl = ImageSequenceClip(imgs, fps=20)
            cl.write_gif(filename, fps=20)
        
    print(f"Total success: {total_succ}/{n_trajs}")
    if use_opencv_to_render:
        cv2.destroyAllWindows()
