import os
from omegaconf import OmegaConf
import numpy as np
from moviepy.editor import ImageSequenceClip

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_mocap import SawyerPickAndPlaceMultiTaskEnvV2
from r3meval.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy


if __name__ == "__main__":
    env_config = {"target_object": "block",
                "skill": "Stack"}
    
    env = SawyerPickAndPlaceMultiTaskEnvV2(env_config)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    save_gif = True

    for obj in env.objects:
        env_config = {"target_object": obj,
            "skill": "Stack"}
    
        env = SawyerPickAndPlaceMultiTaskEnvV2(env_config)
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        policy = PickPlaceMjAbrMocapPolicy(skill=env_config['skill'], target_object=env_config['target_object'])
        step = 0
        
        env.reset_model()
        obs = env.reset()
        img = env.sim.render(height=256,width=256,camera_name="left_cap2")
        img = img[::-1, :, :]
        success = False
        imgs = [img]

        # while (not success) and (step<500):
        while (step<200):
            # a = np.random.uniform(low=0.0, high=10.0, size=4)
            a = policy.get_action(obs)
            next_obs, r, done, info = env.step(a)
            # img = env.get_image()
            img = env.sim.render(height=256,width=256,camera_name="left_cap2")
            img = img[::-1, :, :]
            imgs.append(img)
            obs = next_obs.copy()
            step += 1
            if info['success']:
                success = info['success']
                print(f"================SUCCESS at step: {step}===============")
        if save_gif:
            print("saving gif")
            filename = f'./media/stack_{obj}.gif'
            cl = ImageSequenceClip(imgs, fps=20)
            cl.write_gif(filename, fps=20)
