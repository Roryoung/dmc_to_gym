import sys

from .wrappers import DMCWrapper

class mod_call:
    def __call__(self, env, **kwargs):
        return DMCWrapper(env, **kwargs)

sys.modules[__name__] = mod_call()
