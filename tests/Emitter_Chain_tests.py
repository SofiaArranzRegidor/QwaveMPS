#%% Imports
import QwaveMPS as qmps
import QwaveMPS.operators as qops
from QwaveMPS.symmetrical_coupling_helper import Symmetrical_Coupling_Helper
import numpy as np
import scipy as sci
from matplotlib import rc

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%% Benchmark against other symmetrical method (If symmetrical, need N<=4, M_max<2)
N = 4
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.1#0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1]; taus = [1]*(N-1)
phases = np.zeros(len(taus))

photon_num = 1
d_t_1 = photon_num+1
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 10,#15,
    d_sys_total = d_sys_total,
    d_t_total = [d_t_1]*2,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=64#24
)
tmax = params.tmax
tlist=np.arange(0,tmax+params.delta_t, params.delta_t)
help_obj = Symmetrical_Coupling_Helper(d_sys_total)
l_list = Symmetrical_Coupling_Helper.calc_l_list(taus,d_sys_total,delta_t)
help_obj.set_fback_subchain_lengths(l_list)

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin
i_s0[:,1,:] = 1


# WG
pulse_time = 2
pulse_env = qmps.tophat_envelope(pulse_time, params)
i_n0 = None
#i_n0 = qmps.fock_pulse([pulse_env, pulse_env], pulse_time, params, [photon_num,photon_num])

gamma_ls = [0.5]*N; gamma_rs = [0.5]*N
#%%%
params.bond_max = 36
hams = qmps.hamiltonian_Ntls_sym_eff(params, gamma_ls, gamma_rs, phases=phases)
bins_eff = qmps.t_evol_nmar_sym(hams, i_s0, i_n0, taus, params)

#%%
params.bond_max = 64
hams_ineff = qmps.hamN2LS(params, gamma_ls, gamma_rs, phases=phases)
bins_ineff = qmps.t_evol_nmar(hams_ineff, i_s0, i_n0, taus, params)
# %%
params.bond_max = 64
sys_pop_ops_ineff = []
flux_l_op = qops.b_pop_l(params)
flux_r_op = qops.b_pop_r(params)
flux_op = qmps.b_pop_l(params) + qmps.b_pop_r(params)
sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
for i in range(N):
    sys_pop_ops_ineff.append(qops.extend_op(sys_pop_op, params.d_sys_total, i, reverse_dims=True))

out_bins_ineff = bins_ineff.output_field_states[-1]
out_flux_ineff = qmps.single_time_expectation(out_bins_ineff, flux_op)


sys_pops_ineff = qmps.single_time_expectation(bins_ineff.system_states, sys_pop_ops_ineff)
sys_pops_eff = []
for i in range(len(d_sys_total)):
    sys_pops_eff.append(qops.single_time_expectation(bins_eff.system_states[i], sys_pop_op))

out_bins_eff = bins_eff.output_field_states[-1]
out_flux_eff = qmps.single_time_expectation(out_bins_eff, flux_op)

flux_l_ineff = []
flux_l_eff = []
flux_r_ineff = []
flux_r_eff = []

for i in range(N):
    flux_r_ineff.append(qops.single_time_expectation(bins_ineff.output_field_states[i], flux_r_op))
    flux_r_eff.append(qops.single_time_expectation(bins_eff.output_field_states[i], flux_r_op))
    
    flux_l_ineff.append(qops.single_time_expectation(bins_ineff.output_field_states[-1-i], flux_l_op))
    flux_l_eff.append(qops.single_time_expectation(bins_eff.output_field_states[-1-i], flux_l_op))


#%%
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(sys_pops_eff[i]), color=colors[i], linewidth = 3,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(sys_pops_ineff[i]), color=colors[i], markersize=5, markevery=5,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient')
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

#%%
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_r_eff[i]), color=colors[i], linewidth = 3,linestyle='-',label=r'$n_{\rm R}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(flux_r_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient')
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_l_eff[i]), color=colors[i], linewidth = 3,linestyle='-',label=r'$n_{\rm L}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(flux_l_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient')
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()


