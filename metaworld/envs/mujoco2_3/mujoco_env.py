import abc
import warnings

import glfw
from gym import error
from gym.utils import seeding
import numpy as np
from os import path
import gym

from omegaconf import DictConfig
from pathlib import Path

from robosuite.models import MujocoWorldBase
from robosuite.models.robots import SawyerWithGripper
from robosuite.models.arenas import MultiTaskArena
from robosuite.models.objects.utils import get_obj_from_name

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500

class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_desc, frame_skip):
        if isinstance(model_desc, Path):
            if not path.exists(model_desc):
                raise IOError("File %s does not exist" % model_desc)
            self.model = mujoco.MjModel.from_xml_path(model_desc)
        elif isinstance(model_desc, list):
            self.model = self.create_model_procedural(model_desc)
        else:
            raise NotImplementedError('Model type not supported')

        self.frame_skip = frame_skip
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._did_see_sim_exception = False

        self.renderer = mujoco.Renderer(self.model, 256, 256)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_model_procedural(self, object_list):
        world = MujocoWorldBase()
        mujoco_robot = SawyerWithGripper()
        mujoco_robot.set_base_xpos([0, 0, 0])
        world.merge(mujoco_robot)

        # Merging table arena
        mujoco_arena = MultiTaskArena()
        world.merge(mujoco_arena)
        
        for object_name in object_list:
            obj = get_obj_from_name(object_name)
            obj_body = obj.get_obj()
            world.worldbody.append(obj_body)
            if 'block' not in object_name:
                # obj_body.set('pos', '-0.2 0.45 0.03')
                world.merge_assets(obj)

        world.root.find('compiler').set('inertiagrouprange', '0 5')
        world.root.find('compiler').set('inertiafromgeom', 'auto')
        model = world.get_model(mode="mujoco")
        return model
    
    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def get_state(self):
        """Return MjSimState instance for current state."""
        return  self.data.time, np.copy(self.data.qpos), np.copy(self.data.qvel), np.copy(self.data.act)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.get_state()
        self.data.time = old_state[0]
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.act[:] = np.copy(old_state[3])
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                mujoco.mj_step(self.model, self.data)
            except mujoco.MujocoException as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(self, offscreen=False, camera_name="corner2", resolution=(640, 480)):
        # assert_string = ("camera_name should be one of ",
        #         "corner3, corner, corner2, topview, gripperPOV, behindGripper")
        # assert camera_name in {"corner3", "corner", "corner2", 
        #     "topview", "gripperPOV", "behindGripper"}, assert_string
        if not offscreen:
            self._get_viewer('human').render()
        else:
            # renderer = mujoco.Renderer(self.model, resolution[0], resolution[1])
            self.renderer.update_scene(self.data, camera=camera_name)
            return self.renderer.render()
            # return self.sim.render(
            #     *resolution,
            #     mode='offscreen',
            #     camera_name=camera_name
            # )
    
    def render_new(self, mode='human'):
        if mode == 'human':
            self._get_viewer(mode).render()
        elif mode == 'rgb_array':
            return self.sim.render(
                *self._rgb_array_res,
                mode='offscreen',
                camera_name='topview'
            )[:, :, ::-1]
        else:
            raise ValueError("mode can only be either 'human' or 'rgb_array'")

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xipos[bid].copy()
    
    def get_body_quat(self, body_name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xquat[bid].copy()
