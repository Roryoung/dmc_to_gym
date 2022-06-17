from abc import ABC

class Base_Callback(ABC):

    def __init__(self):
        super(Base_Callback, self).__init__()

    def init_callback(self):
        self._init_callback()

    def _init_callback(self):
        pass

    def augment_reward(self, prev_state, action, next_state, reward, env):
        augmented_reward = self._augment_reward(prev_state, action, next_state, reward, env)

        if isinstance(augmented_reward, type(reward)):
            return augmented_reward
        else:
            return reward

    
    def _augment_reward(self, prev_state, action, next_state, reward, env):
        return reward

# TODO: add chain callback (see stable_baselines3.common.callbacks)