from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.wrappers.vis2d import Vis2dWrapper


def vis2d_nroom_rf(nrooms, memory_size, remove_walls=False, room_size=5):
    env = NRoom(nrooms=nrooms,
                reward_free=True,
                array_observation=True,
                remove_walls=remove_walls,
                room_size=room_size)
    env = Vis2dWrapper(env, n_bins_obs=env.ncols+1, memory_size=memory_size)
    return env
