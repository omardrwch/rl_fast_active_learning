from rlberry.experiment import load_experiment_results
from rlberry.stats import plot_fit_info
import matplotlib.pyplot as plt

import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 6, 5
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 16
matplotlib.rcParams.update({'errorbar.capsize': 0})
# matplotlib.rcParams['text.usetex'] = True


# ------------------------------------------
# Load results
# ------------------------------------------
EXPERIMENT_NAME = 'experiment'
PLOT_TITLES = {
    'rf_express_no_clip': 'RF-Express',
    'rf_ucrl_no_clip': 'RF-UCRL'
}


output_data = load_experiment_results('results', EXPERIMENT_NAME)



# number of video frames, when video is available,
# and description for each action of the env (for video)
N_FRAMES = 200
IMG_SIZE = (25, 5)
ACTION_DESCRIPTION = ['left', 'right', 'down', 'up']


# Get list of AgentStats
_stats_list = list(output_data['stats'].values())
stats_list = []
for stats in _stats_list:
    if stats.agent_name in PLOT_TITLES:
        stats.agent_name = PLOT_TITLES[stats.agent_name]
        stats_list.append(stats)


# -------------------------------
# Plot and save
# -------------------------------
plot_fit_info(stats_list, 'n_visited_states', show=False)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


plot_fit_info(stats_list, 'entropy', show=False)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel("episode", labelpad=0)


plot_fit_info(stats_list, 'error_bound', show=False)
plt.xscale('log')
plt.yscale('log')

# show save all figs
figs = [plt.figure(n) for n in plt.get_fignums()]
for ii, fig in enumerate(figs):
    fname = output_data['experiment_dirs'][0] / 'fig_{}.pdf'.format(ii)
    fig.savefig(fname, format='pdf', bbox_inches='tight')

# save video, if available
for stat in stats_list:
    agent = stat.fitted_agents[0]
    try:
        n_episodes = stat.init_kwargs['n_episodes']
        n_skip = n_episodes // N_FRAMES
        agent.env.plot_trajectories(
            dot_size_means='total_visits',
            n_skip=n_skip,
            dot_scale_factor=10,
            show=False,
            fignum=stat.agent_name)
    except Exception:
        pass
    
plt.show()
