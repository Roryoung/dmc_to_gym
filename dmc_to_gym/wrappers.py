from gym import core, spaces
from dm_env import specs
import numpy as np


def _spec_to_box(spec):
    shape = spec.shape
    if type(spec) == specs.Array:
        high = np.inf * np.ones(shape, dtype=np.float64)
        low = - high    

    elif type(spec) == specs.BoundedArray:
        high = spec.maximum
        low = spec.minimum

    return spaces.Box(low, high, shape=shape, dtype=np.float64)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs:
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
        self,
        dmc_env,
        seed=None,
        from_pixels=False,
        height=480,
        width=640,
        camera_id=[1],
        frame_skip=1,
        channels_first=True
    ):
        self.dmc_env = dmc_env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._render_camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        self.constraints = dmc_env.task.constraints or []
        self.metadata["render.modes"].append("rgb_array")
        self.render_reward = False
        self.render_constraints = False

        # create action
        self._action_space = _spec_to_box(self.dmc_env.action_spec())

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            self._observation_space = _spec_to_box(self.dmc_env.observation_spec())
            
        self._state_space = _spec_to_box(self.dmc_env.observation_spec())
        
        self.current_state = None

        # set seed
        if seed is not None:
            self.seed(seed=seed)


    def __getattr__(self, name):
        return getattr(self.dmc_env, name)


    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                render_colors=False,
                height=self._height,
                width=self._width,
                camera_ids=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()

        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def constraints_space(self):
        return self.constraints

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)


    def step(self, action, render=False):
        action = np.array(action)
        assert self._action_space.contains(action)
        reward = 0
        extra = {"internal_state": self.dmc_env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self.dmc_env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()

            if done:
                break
        
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        self.current_time_step = time_step
        # self._set_reward_colors(reward, time_step.observation[-len(self.constraints):])
        extra["discount"] = time_step.discount
        return obs, reward, done, extra


    def reset(self):
        time_step = self.dmc_env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs


    def render(self, render_colors=True, mode="rgb_array", height=None, width=None, camera_ids=None):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_ids = camera_ids or self._render_camera_id or self._camera_id

        if not render_colors:
            self._set_reward_colors(0, [])
        else:
            self._set_reward_colors(self.current_time_step.reward, self.current_time_step.observation[-len(self.constraints):])

        images = [
            self.dmc_env.physics.render(height=height, width=width, camera_id=camera_id) for camera_id in camera_ids
        ]

        images = np.concatenate(images, axis=1)

        return images


    def render_colors(self, reward=False, constraints=False):
        self.render_reward = reward
        self.render_constraints = constraints


    def render_angles(self, render_camera_ids=None):
        self._render_camera_id = render_camera_ids


    def _set_reward_colors(self, reward, constraints):
        """Sets the highlight, effector and target colors according to the reward."""
        _MATERIALS = ["self", "effector", "target"]
        _DEFAULT = [name + "_default" for name in _MATERIALS]
        _HIGHLIGHT = [name + "_highlight" for name in _MATERIALS]

        if self.render_reward:
            assert 0.0 <= reward <= 1.0
            colors = self.dmc_env.physics.named.model.mat_rgba
            default = colors[_DEFAULT]
            highlight = colors[_HIGHLIGHT]
            blend_coef = reward ** 4  # Better color distinction near high rewards.
            colors[_MATERIALS] = blend_coef * highlight + (1.0 - blend_coef) * default

        if self.render_constraints:
            if len(self.constraints) > 0 and np.sum(constraints) > 0:
                colors[_MATERIALS] = [1,0,0,1]

