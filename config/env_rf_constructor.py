from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom, get_nroom_state_coord
from rlberry.wrappers.vis2d import Vis2dWrapper


def vis2d_nroom_rf(nrooms, memory_size):
    env = NRoom(nrooms=nrooms,
                reward_free=False,
                array_observation=False,
                remove_walls=False,
                room_size=5)
    env = Vis2dWrapper(env,
                       n_bins_obs=env.ncols+1,
                       memory_size=memory_size,
                       state_preprocess_fn=get_nroom_state_coord)
    return env
