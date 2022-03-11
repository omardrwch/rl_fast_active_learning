from rlberry.experiment import load_experiment_results
from evaluation import plot_writer_data
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
_managers_list = list(output_data['manager'].values())
managers_list = []
for manager in _managers_list:
    if manager.agent_name in PLOT_TITLES:
        manager.agent_name = PLOT_TITLES[manager.agent_name]
        managers_list.append(manager)


# -------------------------------
# Plot and save
# -------------------------------
plot_writer_data(managers_list, tag='n_visited_states', show=False)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


plot_writer_data(managers_list, tag='entropy', show=False)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.xlabel("episode", labelpad=0)


plot_writer_data(managers_list, tag='error_bound', show=False)
plt.xscale('log')
plt.yscale('log')

# show save all figs
figs = [plt.figure(n) for n in plt.get_fignums()]
for ii, fig in enumerate(figs):
    fname = output_data['experiment_dirs'][0] / 'fig_{}.pdf'.format(ii)
    fig.savefig(fname, format='pdf', bbox_inches='tight')

plt.show()
