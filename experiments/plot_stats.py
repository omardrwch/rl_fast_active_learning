import os
from rlberry.stats import AgentStats, plot_episode_rewards, compare_policies
import matplotlib.pyplot as plt


# ------------------------------------------
# Parameters
# ------------------------------------------
EXPERIMENT_NAME = 'exp_nroom_large_h'
AGENT_NAME = None

# number of MC simulations for policy eval
N_SIM = 20

# number of video frames, when video is available,
# and description for each action of the env (for video)
N_FRAMES = 200
ACTION_DESCRIPTION = ['left', 'right', 'up', 'down']

RESULTS_DIR = os.path.join('results', EXPERIMENT_NAME)


# Subdirectories with data for each agent
SUBDIRS = [os.path.join(RESULTS_DIR, o)
           for o in os.listdir(RESULTS_DIR)
           if os.path.isdir(os.path.join(RESULTS_DIR, o))]

# Load agent stats from each subdir
stats_list = []
for dd in SUBDIRS:
    fname = os.path.join(dd, 'stats.pickle')
    if (AGENT_NAME is not None) and (AGENT_NAME not in fname):
        continue
    print(fname)
    stats = AgentStats.load(fname)
    stats_list.append(stats)


# -------------------------------
# Plot and save
# -------------------------------
plot_episode_rewards(stats_list, cumulative=False, show=False)
plot_episode_rewards(stats_list, cumulative=True, show=False)
compare_policies(stats_list, n_sim=N_SIM, show=False)

# show save all figs
figs = [plt.figure(n) for n in plt.get_fignums()]
for ii, fig in enumerate(figs):
    fname = os.path.join(RESULTS_DIR, 'fig_{}.pdf'.format(ii))
    fig.savefig(fname, format='pdf')
plt.show()

# save video, if available
for stat in stats_list:
    agent = stat.fitted_agents[0]
    try:
        n_episodes = stat.init_kwargs['n_episodes']
        n_skip = n_episodes // N_FRAMES
        agent.env.plot_trajectories(
            video_filename=os.path.join(RESULTS_DIR, '{}_vid_trajectories.mp4'.format(stat.agent_name)),
            n_skip=n_skip,
            dot_scale_factor=10,
            show=False)
        agent.env.plot_trajectory_actions(
            video_filename=os.path.join(RESULTS_DIR, '{}_vid_actions.mp4'.format(stat.agent_name)),
            action_description=ACTION_DESCRIPTION,
            n_skip=n_skip,
            show=False)
    except AttributeError:
        pass