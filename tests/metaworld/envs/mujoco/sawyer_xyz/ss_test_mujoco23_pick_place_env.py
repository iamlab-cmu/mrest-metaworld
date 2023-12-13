import numpy as np
from moviepy.editor import ImageSequenceClip

try:
    import cv2
except:
    print('OpenCV not found')

from metaworld.envs.mujoco2_3.sawyer_xyz.sawyer_pick_place_multitask_scanned_objs_mocap import SawyerPickAndPlaceMultiTaskGenEnvV23
from r3meval.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy


if __name__ == "__main__":
    
    env_config = {
        "target_object": "bottle_sauce", 
        "stack_on_object": "blockA", 
        "skill": "pick", 
        "task": "pick",
        "only_use_block_objects": False}

    env = SawyerPickAndPlaceMultiTaskGenEnvV23(env_config, data_collection=True)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    use_opencv_to_render = False
    save_gif = True

    n_trajs = 1
    total_succ = 0
    H, W = 256, 256
    cam_name = 'left_cap2'
    for i in range(n_trajs):
    
        env = SawyerPickAndPlaceMultiTaskGenEnvV23(env_config, data_collection=True)
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        step = 0
        
        policy = PickPlaceMjAbrMocapPolicy(skill=env_config['skill'], target_object=env_config['target_object'])

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
