import abc
import warnings

import numpy as np


def assert_fully_parsed(func):
    """Decorator function to ensure observations are fully parsed

    Args:
        func (Callable): The function to check

    Returns:
        (Callable): The input function, decorated to assert full parsing
    """
    def inner(obs):
        obs_dict = func(obs)
        assert len(obs) == sum(
            [len(i) if isinstance(i, np.ndarray) else 1 for i in obs_dict.values()]
        ), 'Observation not fully parsed'
        return obs_dict
    return inner


def move(from_xyz, to_xyz, p):
    """Computes action components that help move from 1 position to another

    Args:
        from_xyz (np.ndarray): The coordinates to move from (usually current position)
        to_xyz (np.ndarray): The coordinates to move to
        p (float): constant to scale response

    Returns:
        (np.ndarray): Response that will decrease abs(to_xyz - from_xyz)

    """
    error = to_xyz - from_xyz
    response = p * error

    if np.any(np.absolute(response) > 1.):
        warnings.warn('Constant(s) may be too high. Environments clip response to [-1, 1]')

    return response


class Policy(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def _parse_obs(obs):
        """Pulls pertinent information out of observation and places in a dict.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            dict: Dictionary which contains information from the observation
        """
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        """Gets an action in response to an observation.

        Args:
            obs (np.ndarray): Observation which conforms to env.observation_space

        Returns:
            np.ndarray: Array (usually 4 elements) representing the action to take
        """
        pass


class MoveTo():
    def __init__(self, target_func, gripper_state, time_limit=None, error_limit=None, p=25.):
        assert time_limit is not None or error_limit is not None, "at least 1 of time limit or error limit should be indicated"

        super().__init__()
        self.target_func = target_func
        self.gripper_state = gripper_state
        self.time_limit = time_limit
        self.error_limit = error_limit
        self.step_in_stage = 0
        self.p = p

    def ends(self, state):
        # calculate error
        ee_xyz = state['ee_xyz']
        error = np.linalg.norm(ee_xyz - self.target_func(state)[:3])
        end = False
        # whether stop criterion has been reached
        if self.time_limit is not None:
            if self.step_in_stage >= self.time_limit:
                end = True
        if self.error_limit is not None:
            if error <= self.error_limit:
                end = True
        return end

    def plan_action(self, state):
        self.step_in_stage += 1

        action = np.zeros((4))
        error = self.target_func(state)[:3] - state["ee_xyz"]
        action[:3] = self.p * error

        # Set gripper force
        if self.gripper_state is not None:
            action = gripper_control_func(action, self.gripper_state)

        end = self.ends(state)
        return action, end

def gripper_control_func(action, gripper_state):
    if 'close' in gripper_state:
        action[-1] = 1.
    else:
        action[-1] = -1.
    return action