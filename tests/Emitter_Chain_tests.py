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

#%% Benchmark against other symmetrical method (need N<=4, M_max<=1 if N=4)
N = 2#4
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.1#0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1]; taus = [1]*(N-1)
phases = np.zeros(len(taus))
photon_num = 1
d_t_1 = photon_num+1
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 1,#10,
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
params.bond_max = 64
hams_ineff = qmps.hamN2LS(params, gamma_ls, gamma_rs, phases=phases)
bins_ineff = qmps.t_evol_nmar(hams_ineff, i_s0, i_n0, taus, params)
#%%% Calculate Observables
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

ax.plot(tlist,np.real(total_quanta_eff/total_quanta_eff[0]), color=colors[N], linewidth = 2.5,linestyle='-',label=r'$\frac{n_{\rm tot}}{n_{\rm tot}}$')
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
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient'),
    Line2D([0], [0], color='black', marker='^', linestyle='None', label='Eff. Single Bin'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

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
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
   Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

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
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient')
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

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
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])

plt.minorticks_on()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(which='minor', color='gray', linestyle=':', alpha=0.5)

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

sim_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Efficient'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Inefficient')
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

#%% Benchmark against other Chiral method (more versatile N cases, can go higher)
N = 4
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.5#0.05
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
    bond_max=36#24
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
flux_r_op = qops.b_pop_l(params)

sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
for i in range(N):
    sys_pop_ops.append(qops.extend_op(sys_pop_op, params.d_sys_total, i, reverse_dims=True))

sys_pops_sym = []
sys_pops_chiral = []
sys_pops_chiral_single_bin = qmps.single_time_expectation(total_sys_bins_chiral, sys_pop_ops)
sys_pops_sym_single_bin = qmps.single_time_expectation(total_sys_bins, sys_pop_ops)

for i in range(len(d_sys_total)):
    sys_pops_eff.append(qops.single_time_expectation(bins_sym.system_states[i], sys_pop_op))
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

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(sys_pops_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(sys_pops_chiral[i]), color=colors[i], markersize=5, markevery=slice(0,None,5),linestyle='None', marker='o')#, markeredgecolor='black', alpha=0.5)
    ax.plot(tlist,np.real(sys_pops_sym_single_bin[i]), color=colors[i], markersize=5, markevery=slice(2,None,5),linestyle='None', marker='^')#, markeredgecolor='black', alpha=0.5)

ax.plot(tlist,np.real(total_quanta_sym/total_quanta_sym[0]), color=colors[N], linewidth = 2.5,linestyle='-',label=r'$\frac{n_{\rm tot}}{n_{\rm tot}}$')
ax.plot(tlist,np.real(total_quanta_chiral/total_quanta_chiral[0]), color=colors[N], markersize=5, markevery=5,linestyle='None', marker='o')


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
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Sym.'),
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Chiral'),
    Line2D([0], [0], color='black', marker='^', linestyle='None', label='Sym. Single Bin'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

# Right propagating flux plots

fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N):
    ax.plot(tlist,np.real(flux_r_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(flux_r_chiral[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')


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
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Sym.'),
   Line2D([0], [0], color='black', marker='o', linestyle='None', label='Chiral'),
]

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()

# Left propagating flux plots

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

# Right Loop Quanta
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N-1):
    ax.plot(tlist,np.real(loop_integrated_r_sym[i]), color=colors[i], linewidth = 2.5,linestyle='-',label=r'$n_{\rm R,loop}^{('+str(i)+r')}$')
    ax.plot(tlist,np.real(loop_integrated_r_chiral[i]), color=colors[i], markersize=5, markevery=0.05,linestyle='None', marker='o')
    
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

legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.75),labelspacing=0.2, ncol=1)
ax.add_artist(legend1)

legend2 = ax.legend(handles=sim_handles, title="Simulation", handlelength=0.6,
                    loc='center left', bbox_to_anchor=(1.02, 0.25), ncol=1)

plt.tight_layout()
plt.show()
# %%
