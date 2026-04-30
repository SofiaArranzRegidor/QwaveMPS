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
import os
from pathlib import Path
from matplotlib.colors import Normalize
from numpy.linalg import matrix_power

save_dir = Path('/home/mattkozma/Documents/QWaveMPS_Scripts/data/emitterCascade/benchmarks')
save_flag = True

#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%% Benchmark against other symmetrical method (need N<=4, M_max<=1 if N=4)
#%%% Setup simulation parameters
N = 4
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.1#0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1]; taus = [1]*(N-1)
phases = np.zeros(len(taus))
photon_num = 1
d_t_1 = photon_num+1
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 10,
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
i_s0[:,3,:] = 1

# WG
pulse_time = 2
pulse_env = qmps.tophat_envelope(pulse_time, params)
i_n0 = None
#i_n0 = qmps.fock_pulse([pulse_env, pulse_env], pulse_time, params, [photon_num,photon_num])

gamma_ls = [0.5]*N; gamma_rs = [0.5]*N
#%%% Efficient Simulation
params.bond_max = 36
hams = qmps.hamiltonian_Ntls_sym_eff(params, gamma_ls, gamma_rs, phases=phases)
bins_eff,total_sys_bins,_ = qmps.t_evol_nmar_sym(hams, i_s0, i_n0, taus, params, store_total_sys_flag=True)

#%%% Inefficient Simulation
params.bond_max = 48
hams_ineff = qmps.hamN2LS(params, gamma_ls, gamma_rs, phases=phases)
bins_ineff = qmps.t_evol_nmar(hams_ineff, i_s0, i_n0, taus, params)
#%%% Calculate Observables
params.bond_max = 48
sys_pop_ops_ineff = []
flux_l_op = qops.b_pop_l(params)
flux_r_op = qops.b_pop_r(params)
flux_op = qmps.b_pop_l(params) + qmps.b_pop_r(params)
sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
for i in range(N):
    sys_pop_ops_ineff.append(qops.extend_op(sys_pop_op, params.d_sys_total, i, reverse_dims=True))

out_bins_ineff = bins_ineff.output_field_states[-1]
out_flux_ineff = qmps.single_time_expectation(out_bins_ineff, flux_op)


sys_pops_eff = []
sys_pops_eff_single_bin = []
sys_pops_ineff = qmps.single_time_expectation(bins_ineff.system_states, sys_pop_ops_ineff)
sys_pops_eff_single_bin = qmps.single_time_expectation(total_sys_bins, sys_pop_ops_ineff)

for i in range(len(d_sys_total)):
    sys_pops_eff.append(qops.single_time_expectation(bins_eff.system_states[i], sys_pop_op))

out_bins_eff = bins_eff.output_field_states[-1]
out_flux_eff = qmps.single_time_expectation(out_bins_eff, flux_op)

flux_l_ineff = []
flux_l_eff = []
flux_r_ineff = []
flux_r_eff = []

loop_integrated_l_ineff = []
loop_integrated_l_eff = []
loop_integrated_r_ineff = []
loop_integrated_r_eff = []


for i in range(N):
    flux_r_ineff.append(qops.single_time_expectation(bins_ineff.output_field_states[i], flux_r_op))
    flux_r_eff.append(qops.single_time_expectation(bins_eff.output_field_states[i], flux_r_op))
    
    flux_l_ineff.append(qops.single_time_expectation(bins_ineff.output_field_states[-1-i], flux_l_op))
    flux_l_eff.append(qops.single_time_expectation(bins_eff.output_field_states[-1-i], flux_l_op))
    
    if i < N-1:
        loop_integrated_r_ineff.append(qops.loop_integrated_statistics(flux_r_ineff[i], params, taus[i]))
        loop_integrated_r_eff.append(qops.loop_integrated_statistics(flux_r_eff[i], params, taus[i]))
    
    if i > 0:
        loop_integrated_l_ineff.append(qops.loop_integrated_statistics(flux_l_ineff[i], params, taus[i-1]))
        loop_integrated_l_eff.append(qops.loop_integrated_statistics(flux_l_eff[i], params, taus[i-1]))

total_sys_pop_ineff = np.sum(sys_pops_ineff, axis=0)
total_field_ineff = np.sum(loop_integrated_r_ineff, axis=0) + np.sum(loop_integrated_l_ineff, axis=0) \
                    + np.cumsum(flux_l_ineff[0] + flux_r_ineff[-1]) * delta_t
total_quanta_ineff = total_sys_pop_ineff + total_field_ineff

total_sys_pop_eff = np.sum(sys_pops_ineff, axis=0)
total_field_eff = np.sum(loop_integrated_r_eff, axis=0) + np.sum(loop_integrated_l_eff, axis=0) \
                    + np.cumsum(flux_l_eff[0] + flux_r_eff[-1]) * delta_t
total_quanta_eff = total_sys_pop_eff + total_field_eff


#%%% Plots
#===============================================================================================
# Population plots
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(sys_pops_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(sys_pops_ineff[i]), color=colors[i], markersize=5, markevery=slice(0,None,5),linestyle='None', marker='o')#, markeredgecolor='black', alpha=0.5)
    ax.plot(tlist,np.real(sys_pops_eff_single_bin[i]), color=colors[i], markersize=5, markevery=slice(2,None,5),linestyle='None', marker='^')#, markeredgecolor='black', alpha=0.5)

ax.plot(tlist,np.real(total_quanta_eff/total_quanta_eff[0]), color=colors[N], linewidth = 2.5,linestyle='-',label=r'$\frac{n_{\rm tot}}{m}$')
ax.plot(tlist,np.real(total_quanta_ineff/total_quanta_eff[0]), color=colors[N], markersize=5, markevery=5,linestyle='None', marker='o')


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
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient'),
    Line2D([0], [0], color='black', marker='^', linestyle='None', label='Eff. 1 Bin'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'pops.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#===============================================================================================
# Right propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_r_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(flux_r_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
   Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightFlux.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Left propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_l_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm L}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(flux_l_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'leftFlux.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Left Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_l_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm L,loop}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(loop_integrated_l_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')

plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'leftLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Right Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R,loop}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(loop_integrated_r_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')
    
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#===============================================================================================
# Total Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r_eff[i] + loop_integrated_l_eff[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm loop}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(loop_integrated_r_ineff[i] + loop_integrated_l_ineff[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')
    
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'totalLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#%%% Calculation of second order correlations
a_ops = []; b_ops = []; c_ops = []; d_ops = []
b_dag_r = qops.b_dag_r(params) ; b_r = qops.b_r(params)
b_dag_l = qops.b_dag_l(params); b_l = qops.b_l(params)
dim = b_r.shape[0]

# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(b_r)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(b_l)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


"""Also calculate the G2"""
# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(b_dag_r)
c_ops.append(b_r)
d_ops.append(b_r)


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(b_dag_l)
c_ops.append(b_l)
d_ops.append(b_l)

"""G2_LR"""
X = b_dag_r + b_r
a_ops.append(b_dag_l)
b_ops.append(b_dag_r)
c_ops.append(b_r)
d_ops.append(b_l)
"""G2_RL"""
X = b_dag_r + b_r
a_ops.append(b_dag_r)
b_ops.append(b_dag_l)
c_ops.append(b_l)
d_ops.append(b_r)

corrs_eff, corr_tlist = qmps.correlation_4op_2t(bins_eff.correlation_bins, a_ops, b_ops, c_ops, d_ops, params)
corrs_ineff, corr_tlist = qmps.correlation_4op_2t(bins_ineff.correlation_bins, a_ops, b_ops, c_ops, d_ops, params)

padding_factor = 0
padding = corrs_eff[0].shape[0]*padding_factor

spect_intensity_r_eff, w_list_intensity = qmps.spectral_intensity(corrs_eff[0], params, padding=padding)
spect_intensity_l_eff, w_list_intensity = qmps.spectral_intensity(corrs_eff[1], params, padding=padding)
spect_intensity_r_ineff, w_list_intensity = qmps.spectral_intensity(corrs_ineff[0], params, padding=padding)
spect_intensity_l_ineff, w_list_intensity = qmps.spectral_intensity(corrs_ineff[1], params, padding=padding)

time_dep_spec_r_eff, w_list_spectrum = qmps.time_dependent_spectrum(corrs_eff[0], params, padding=padding)
time_dep_spec_l_eff, w_list_spectrum = qmps.time_dependent_spectrum(corrs_eff[1], params, padding=padding)
time_dep_spec_r_ineff, w_list_spectrum = qmps.time_dependent_spectrum(corrs_ineff[0], params, padding=padding)
time_dep_spec_l_ineff, w_list_spectrum = qmps.time_dependent_spectrum(corrs_ineff[1], params, padding=padding)

#%%% Make the 2D correlation plots
'''
Indices:
    0: G_R^1
    1: G_L^1
    2: G_RR^2
    3: G_LL^2
    4: G_LR^2
    5: G_RL^2
'''
X,Y = np.meshgrid(corr_tlist,corr_tlist)
z1 = np.real(qmps.transform_t_tau_to_t1_t2(corrs_eff[3]))
z2 = np.real(qmps.transform_t_tau_to_t1_t2(corrs_ineff[3]))

absMax = max(np.abs(z1).max(),np.abs(z2).max())
norm = Normalize(vmin=-absMax, vmax=absMax)


cmap = 'seismic'
fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
cf1 = axs[0].pcolormesh(X,Y,z1,shading='gouraud',cmap=cmap, norm=norm,rasterized=True)
cf2 = axs[1].pcolormesh(X,Y,z2,shading='gouraud',cmap=cmap, norm=norm,rasterized=True)

cbar = fig.colorbar(cf2,ax=axs[1], pad=0)
cbar.set_label(r'$G_{LL}^{(1)}(t,\tau)\ [\gamma]$',labelpad=0)
cbar.ax.yaxis.set_major_formatter(formatter)
cbar.ax.tick_params(width=0.75)

for i in range(2):
    #axs[i].set_xlabel(r'$(\omega-\omega_p)/\gamma$')
    axs[i].set_xlabel(r'$\gamma(t+\tau)$')
    axs[i].xaxis.set_major_formatter(formatter)
    axs[i].set_xlim([0,tmax])

    axs[i].set_ylim([0,tmax])
    if i == 0:
        axs[i].set_ylabel(r'Time, $\gamma t$')
        axs[i].yaxis.set_major_formatter(formatter)
    else:
        axs[i].set_yticks([])
axs[0].text(0.95, 0.95, "Efficient", transform=axs[0].transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
axs[1].text(0.95, 0.95, "Inefficient", transform=axs[1].transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

if save_flag:
    plt.savefig(save_dir / 'corr.pdf', format='pdf', bbox_inches='tight', dpi=500)

plt.show()

# Spectra stuff
X,Y = np.meshgrid(w_list_spectrum,corr_tlist)
z1 = np.real(spect_intensity_l_eff)
z2 = np.real(spect_intensity_l_ineff)

absMax = max(np.abs(z1).max(),np.abs(z2).max())
norm = Normalize(vmin=-absMax, vmax=absMax)


cmap = 'seismic'
fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
cf1 = axs[0].pcolormesh(X,Y,z1,shading='gouraud',cmap=cmap, norm=norm,rasterized=True)
cf2 = axs[1].pcolormesh(X,Y,z2,shading='gouraud',cmap=cmap, norm=norm,rasterized=True)

cbar = fig.colorbar(cf2,ax=axs[1], pad=0)
cbar.set_label(r'$S_{L}(\omega,t)\ [A.u.]$',labelpad=0)
cbar.ax.yaxis.set_major_formatter(formatter)
cbar.ax.tick_params(width=0.75)

for i in range(2):
    axs[i].set_xlabel(r'$(\omega-\omega_p)/\gamma$')
    axs[i].xaxis.set_major_formatter(formatter)
    axs[i].set_xlim([-10,10])

    axs[i].set_ylim([0,tmax])
    if i == 0:
        axs[i].set_ylabel(r'Time, $\gamma t$')
        axs[i].yaxis.set_major_formatter(formatter)
    else:
        axs[i].set_yticks([])
axs[0].text(0.95, 0.95, "Efficient", transform=axs[0].transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
axs[1].text(0.95, 0.95, "Inefficient", transform=axs[1].transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

if save_flag:
    plt.savefig(save_dir / 'corr_spect.pdf', format='pdf', bbox_inches='tight', dpi=500)

plt.show()


#%% Benchmark against other Chiral method (more versatile N cases, can go higher)
#%%% Setup simulation parameters
N = 6
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.02#0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1]; taus = [0.5]*(N-1)
phases = np.zeros(len(taus))
photon_num = 1
d_t_1 = photon_num+1
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 20,#15,
    d_sys_total = d_sys_total,
    d_t_total = [d_t_1]*2,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=160#24
)
tmax = params.tmax
tlist=np.arange(0,tmax+params.delta_t, params.delta_t)
tlist2 = tlist
help_obj = Symmetrical_Coupling_Helper(d_sys_total)
l_list = Symmetrical_Coupling_Helper.calc_l_list(taus,d_sys_total,delta_t)
help_obj.set_fback_subchain_lengths(l_list)

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin
i_s0[:,np.prod(d_sys_total)-1,:] = 1

# WG
pulse_time = 2
pulse_env = qmps.tophat_envelope(pulse_time, params)
i_n0 = None
#i_n0 = qmps.fock_pulse([pulse_env, pulse_env], pulse_time, params, [photon_num,photon_num])

gamma_rs = [1]*N; gamma_ls = [0]*N
#%%% Efficient Simulation
hams = qmps.hamiltonian_Ntls_sym_eff(params, gamma_ls, gamma_rs, phases=phases)
bins_sym,total_sys_bins,_ = qmps.t_evol_nmar_sym(hams, i_s0, i_n0, taus, params, store_total_sys_flag=True)

#%%% Chiral Simulation
# Could change params.d_t_total to 1D if accounted for in observables and input field too
#params.d_t_total = [2]
hams_chiral = []
for i in range(len(d_sys_total)):
    hm = qmps.hamiltonians.hamiltonian_1tls_chiral(params)
    hams_chiral.append(hm)

bins_chiral,total_sys_bins_chiral,_ = qmps.t_evol_nmar_chiral(hams_chiral, i_s0, i_n0, taus, params, store_total_sys_flag=True)


#%%% Calculate Observables
sys_pop_ops = []
flux_r_op = qops.b_pop_r(params)
flux_l_op = qops.b_pop_l(params)

sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
for i in range(N):
    sys_pop_ops.append(qops.extend_op(sys_pop_op, params.d_sys_total, i, reverse_dims=True))

sys_pops_sym = []
sys_pops_chiral = []
sys_pops_chiral_single_bin = qmps.single_time_expectation(total_sys_bins_chiral, sys_pop_ops)
sys_pops_sym_single_bin = qmps.single_time_expectation(total_sys_bins, sys_pop_ops)

for i in range(len(d_sys_total)):
    sys_pops_sym.append(qops.single_time_expectation(bins_sym.system_states[i], sys_pop_op))
    sys_pops_chiral.append(qops.single_time_expectation(bins_chiral.system_states[i], sys_pop_op))

flux_l_sym = []
flux_r_chiral = []
flux_r_sym = []

loop_integrated_l_sym = []
loop_integrated_r_chiral = []
loop_integrated_r_sym = []


for i in range(N):
    flux_r_chiral.append(qops.single_time_expectation(bins_chiral.output_field_states[i], flux_r_op))
    flux_r_sym.append(qops.single_time_expectation(bins_sym.output_field_states[i], flux_r_op))
    
    flux_l_sym.append(qops.single_time_expectation(bins_sym.output_field_states[-1-i], flux_l_op))
    
    if i < N-1:
        loop_integrated_r_chiral.append(qops.loop_integrated_statistics(flux_r_chiral[i], params, taus[i]))
        loop_integrated_r_sym.append(qops.loop_integrated_statistics(flux_r_sym[i], params, taus[i]))
    
    if i > 0:
        loop_integrated_l_sym.append(qops.loop_integrated_statistics(flux_l_sym[i], params, taus[i-1]))

total_sys_pop_chiral = np.sum(sys_pops_chiral, axis=0)
total_field_chiral = np.sum(loop_integrated_r_chiral, axis=0)\
                    + np.cumsum(flux_r_chiral[-1]) * delta_t
total_quanta_chiral = total_sys_pop_chiral + total_field_chiral

total_sys_pop_sym = np.sum(sys_pops_sym, axis=0)
total_field_sym = np.sum(loop_integrated_r_sym, axis=0) + np.sum(loop_integrated_l_sym, axis=0) \
                    + np.cumsum(flux_l_sym[0] + flux_r_sym[-1]) * delta_t
total_quanta_sym = total_sys_pop_sym + total_field_sym


#%%% Plots
# Population plots
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors
ps = len(tlist)
ps = int(ps/10)

fig, ax = plt.subplots(figsize=(9, 5))

for i in range(N):
    ax.plot(tlist,np.real(sys_pops_sym[i]), color=colors[i%10], linewidth = 2.5,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    ax.plot(tlist2,np.real(sys_pops_chiral[i]), color=colors[i%10], markersize=5, markevery=slice(0,None,ps),linestyle='None', marker='o')#, markeredgecolor='black', alpha=0.5)
    ax.plot(tlist,np.real(sys_pops_sym_single_bin[i]), color=colors[i%10], markersize=5, markevery=slice(4,None,ps),linestyle='None', marker='^')#, markeredgecolor='black', alpha=0.5)
    ax.plot(tlist2,np.real(sys_pops_chiral_single_bin[i]), color=colors[i%10], markersize=5, markevery=slice(8,None,ps),linestyle='None', marker='s')#, markeredgecolor='black', alpha=0.5)

ax.plot(tlist,np.real(total_quanta_sym/total_quanta_sym[0]), color=colors[N%10], linewidth = 2.5,linestyle='--',label=r'$\frac{n_{\rm tot}}{m}$')
ax.plot(tlist2,np.real(total_quanta_chiral/total_quanta_chiral[0]), color=colors[N%10], markersize=5, markevery=0.05,linestyle='None', marker='o')


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
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.05),labelspacing=0.2,ncol=2, columnspacing=0.5)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Sym.'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Chiral'),
    Line2D([0], [0], color='black', marker='^', linestyle='None', label='Sym. 1 Bin'),
    Line2D([0], [0], color='black', marker='s', linestyle='None', label='Ch. 1 Bin'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
            loc='center left', bbox_to_anchor=(1.02, 0.05), ncol=2, columnspacing=0.5)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'pops.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

# Right propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_r_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R}^{('+str(i)+r')}$')
    ax.plot(tlist2,np.real(flux_r_chiral[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1.),labelspacing=0.2,ncol=2, columnspacing=0.5)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Sym.'),
   Line2D([0], [0], color='black', marker='o', linestyle='None', label='Chiral'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6, 
                    loc='center left', bbox_to_anchor=(1.02, 0))

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightFlux.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

# Left propagating flux plots
'''
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(N):
    ax.plot(tlist,np.real(flux_l_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm L}^{('+str(i)+r')}$')

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
plt.tight_layout()
plt.show()
'''

# Right Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R,loop}^{('+str(i)+r')}$')
    ax.plot(tlist2,np.real(loop_integrated_r_chiral[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')
    
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1.),labelspacing=0.2,ncol=2, columnspacing=0.5)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6, 
                    loc='center left', bbox_to_anchor=(1.02, 0))

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#%% Benchmark Against paper sym results
#%%% Setup simulation parameters
N = 4
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.1#0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1];
tau = 0.4
taus = [tau]*(N-1)
phases = np.zeros(len(taus))
photon_num = 3
d_t_1 = photon_num+1
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 10,
    d_sys_total = d_sys_total,
    d_t_total = [d_t_1]*2,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=128#24
)
tmax = params.tmax
tlist=np.arange(0,tmax+params.delta_t, params.delta_t)
help_obj = Symmetrical_Coupling_Helper(d_sys_total)
l_list = Symmetrical_Coupling_Helper.calc_l_list(taus,d_sys_total,delta_t)
help_obj.set_fback_subchain_lengths(l_list)

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin
i_s0[:,np.prod(d_sys_total)-1,:] = 1

# WG
pulse_time = 2
pulse_env = qmps.tophat_envelope(pulse_time, params)
i_n0 = None
#i_n0 = qmps.fock_pulse([pulse_env, pulse_env], pulse_time, params, [photon_num,photon_num])

gamma_ls = [0.5]*N; gamma_rs = [0.5]*N
#%%%  Simulation
hams = qmps.hamiltonian_Ntls_sym_eff(params, gamma_ls, gamma_rs, phases=phases)
bins,total_sys_bins,_ = qmps.t_evol_nmar_sym(hams, i_s0, i_n0, taus, params, store_total_sys_flag=True)

#%%% Calculate Observables
sys_pop_ops = []
b_l = qops.b_l(params); b_r = qops.b_r(params);
b_l_dag = qops.b_dag_l(params); b_r_dag = qops.b_dag_r(params);


flux_l_op = qops.b_pop_l(params)
flux_r_op = qops.b_pop_r(params)
flux_op = qmps.b_pop_l(params) + qmps.b_pop_r(params)
sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
for i in range(N):
    sys_pop_ops.append(qops.extend_op(sys_pop_op, params.d_sys_total, i, reverse_dims=True))

sys_pops = []
sys_pops_single_bin = qmps.single_time_expectation(total_sys_bins, sys_pop_ops)

for i in range(len(d_sys_total)):
    sys_pops.append(qops.single_time_expectation(bins.system_states[i], sys_pop_op))

out_bins = bins.output_field_states[-1]
out_flux = qmps.single_time_expectation(out_bins, flux_op)

flux_l = []
flux_r = []
flux_bidirectional = []

loop_integrated_l = []
loop_integrated_r = []
loop_integrated_tot = []


for i in range(N):
    flux_r.append(qops.single_time_expectation(bins.output_field_states[i], flux_r_op))
    flux_l.append(qops.single_time_expectation(bins.output_field_states[-1-i], flux_l_op))
    flux_bidirectional.append(qops.single_time_expectation(bins.output_field_states[i], flux_op))
    if i < N-1:
        loop_integrated_r.append(qops.loop_integrated_statistics(flux_r[i], params, taus[i]))
    
    if i > 0:
        loop_integrated_l.append(qops.loop_integrated_statistics(flux_l[i], params, taus[i-1]))
for i in range(N-1):
    loop_integrated_tot.append(loop_integrated_r[i] + loop_integrated_l[i])

total_sys_pop = np.sum(sys_pops, axis=0)
total_field = np.sum(loop_integrated_r, axis=0) + np.sum(loop_integrated_l, axis=0) \
                    + np.cumsum(flux_l[0] + flux_r[-1]) * delta_t
total_quanta = total_sys_pop + total_field

# To the left of emitter 0 only left propagating field
same_time_G_l_ops = []
for i in range(d_t_1):
    same_time_G_l_ops.append(matrix_power(b_l_dag, i)@matrix_power(b_l, i))

same_time_G_l = qops.single_time_expectation(bins.output_field_states[-1], same_time_G_l_ops)
same_time_g_l = []
for i in range(d_t_1):
    same_time_g_l.append(same_time_G_l[i] / flux_l[0]**i)

#%%% Plots (Same as before)
#===============================================================================================
# Population plots
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    line_style = '-'
    if i > (N-1)/2:
        line_style = '--'
    ax.plot(tlist,np.real(sys_pops[i]), color=colors[i], linewidth = 2.5,linestyle=line_style,label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    
    #ax.plot(tlist,np.real(sys_pops_single_bin[i]), color=colors[i], markersize=5, markevery=slice(2,None,5),linestyle='None', marker='^')#, markeredgecolor='black', alpha=0.5)

ax.plot(tlist,np.real(total_quanta/total_quanta[0]), color=colors[N], linewidth = 2.5,linestyle='-',label=r'$\frac{n_{\rm tot}}{m}$')


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
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'pops.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#===============================================================================================
# Right propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_r[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R}^{('+str(i)+r')}$')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),labelspacing=0.2, ncol=1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
]

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightFlux.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Left propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_l[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm L}^{('+str(i)+r')}$')


plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'leftFlux.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Left Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_l[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm L,loop}^{('+str(i)+r')}$')

plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'leftLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#===============================================================================================
# Right Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R,loop}^{('+str(i)+r')}$')
    
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'rightLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#===============================================================================================
# Total Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r[i] + loop_integrated_l[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm loop}^{('+str(i)+r')}$')
    
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,None])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 1),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'totalLoopQuanta.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
#%%% Plots (Reproducing paper plots)
#%%%%
#===============================================================================================
# Fig 2c plots
# Need intermediary two time correlation function for the cross terms in between the emitters
# As need to calculate <a^\dag a>(x) = <(a_L + a_R)^\dag (a_L + a_r)>(x)
# = <a_L^\dag a_L>(x) + <a_R^\dag a_R>(x) + <a_L^\dag a_R>(x) + <a_R^\dag a_L>(x)
# = = <a_L^\dag a_L>(x) + <a_R^\dag a_R>(x) + (<a_L^\dag a_R>(x) + H.c.)
fonts=18
pic_style(fonts)
plt.rcParams['figure.dpi'] = 500
colors = plt.cm.tab10.colors

end_loop_field_quantas = [[] for _ in range(N-1)]
for i in range(N-1):
    end_loop_field_quantas[i] = flux_r[i][-l_list[i]:][::-1] + flux_l[i+1][-l_list[i]:]
    #end_loop_field_quantas[i] = flux_bidirectional[i][-l_list[i]:][::-1]

field_quanta = np.array([])
field_quanta = np.append(field_quanta, flux_l[0])
for i in range(len(end_loop_field_quantas)):
    field_quanta = np.append(field_quanta, end_loop_field_quantas[i])

field_quanta = np.append(field_quanta, flux_r[-1][::-1])

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(np.arange(0,len(field_quanta)*delta_t - delta_t/2,delta_t),np.real(field_quanta), color=colors[0], linewidth = 2.5,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    


plt.xlabel('Time, $\gamma t$')
plt.ylabel(r'$n_E(x)$')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,None])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5, axis='x')

legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9),labelspacing=0.2, ncol=1)

plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'fig2c.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#%%%%
#===============================================================================================
# Fig 3 plots
start_ind = 2
plt2_taumaxes = [4,3.5]
row_num = d_t_1 - start_ind
fig, axs = plt.subplots(row_num, 2, figsize=(12, 4*row_num), constrained_layout=True)
for i,m in enumerate(range(start_ind,d_t_1)):
    if row_num>1:
        ax = axs[i,0]
    else:
        ax = axs[0]
    ax.plot(tlist/tau,delta_t**(m/2.)*np.real(flux_l[0]**(m)), color=colors[0], linewidth = 2.5,linestyle='-',label=r'$\sqrt{\Delta t}^{'+str(m)+r'}(n_{\rm L}^{(0)})^{'+str(m)+r'}(t)$',markersize=5,marker='o')
    ax.plot(tlist/tau,delta_t**(m/2.)*np.real(same_time_G_l[m]), color=colors[1], linewidth = 2.5,linestyle='--',label=r'$\sqrt{\Delta t}^{'+str(m)+r'}G_{\rm LL}^{('+str(m)+r')}(t,0)$',markersize=5,marker='o')
    ax.set_xlabel(r'Time, $\gamma \tau$')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    formatter = FuncFormatter(clean_ticks)
    ax.set_ylim([0.,None])
    ax.set_xlim([0.,tmax])

    for j in range(0, int(tmax) + 1):
        ax.axvline(j, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9),labelspacing=0.2, ncol=1)
 
    
    plt2_taumax = plt2_taumaxes[i]
    tmax_ind = np.argmax(tlist > plt2_taumax*tau)

    if row_num>1:
        ax = axs[i,1]
    else:
        ax = axs[1]
    ax.plot(tlist[:tmax_ind]/tau,np.real(same_time_g_l[m][:tmax_ind]), color=colors[0], linewidth = 2.5,linestyle='-',label=r'$g_{\rm LL}^{('+str(m)+r')}(t,0)$',markersize=5,marker='o')
    ax.set_xlabel(r'Time, $\gamma \tau$')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    formatter = FuncFormatter(clean_ticks)
    ax.set_ylim([0.,None])
    ax.set_xlim([0.,plt2_taumax])

    for j in range(0, int(tmax) + 1):
        ax.axvline(j, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
    legend1 = ax.legend(loc='upper right', bbox_to_anchor=(1, 0.9),labelspacing=0.2, ncol=1)




plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'fig3.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#%%%%
#===============================================================================================
# Fig 5 plot
import itertools
def kron_all(ops):
    """Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def P_N_m(N, m, site_op=None, other_sites=None):
    """
    Construct the collective operator P(N, m).
    
    Parameters:
        N : int
            Number of sites
        m : int
            Number of excitations (number of ladder ops)
        op_type : str
            "plus" or "minus"
    """
    if site_op is None:
        site_op = qops.sigmaplus() @ qops.sigmaminus()
    if other_sites is None:
        other_sites = qops.sigmaminus() @ qops.sigmaplus()
    
    dim = 2**N
    total = np.zeros((dim, dim), dtype=complex)
    
    # All combinations of m sites out of N
    for sites in itertools.combinations(range(N), m):
        ops = []
        for i in range(N):
            if i in sites:
                ops.append(site_op)
            else:
                ops.append(other_sites)
        
        total += kron_all(ops)
    
    return total

ops = []
for i in range(N+1):
    ops.append(P_N_m(N, i))
    
excitation_num_prob = qops.single_time_expectation(total_sys_bins, ops)

# Graph the darn things
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(N+1):
    ax.plot(tlist,np.real(excitation_num_prob[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$P^{('+str(i)+r')}(t)$',markersize=3,marker='o')

ax.set_xlabel(r'Time, $\gamma \tau$')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
formatter = FuncFormatter(clean_ticks)
ax.set_ylim([0.,1.05])
ax.set_xlim([0.,tmax])

for j in range(0, int(tmax) + 1):
    ax.axvline(j, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.),labelspacing=0.2, ncol=2)


plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'fig5.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()

#%%%%
#===============================================================================================
# Fig 5 plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(tlist,np.real(np.real(np.sum(sys_pops, axis=0))/N), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$P^{('+str(i)+r')}(t)$',markersize=3,marker='o')

ax.set_xlabel(r'Time, $\gamma \tau$')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
formatter = FuncFormatter(clean_ticks)
ax.set_ylim([0.,1.05])
ax.set_xlim([0.,tmax])

for j in range(0, int(tmax) + 1):
    ax.axvline(j, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.),labelspacing=0.2, ncol=2)


plt.tight_layout()
if save_flag:
    plt.savefig(save_dir / 'fig5.pdf', format='pdf', bbox_inches='tight', dpi=500)
plt.show()
