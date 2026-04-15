#%% Imports
import QwaveMPS as qmps
import QwaveMPS.operators as qops
from QwaveMPS.symmetrical_coupling_helper import Symmetrical_Coupling_Helper
from QwaveMPS.simulation import _svd_tensors
import numpy as np
import scipy as sci
from ncon import ncon
import copy
from matplotlib import rc

#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%% Test the separation of the system bin, and it's reordering in preparation of the MPS
N = 8
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 1
taus = [2]*(N-1); taus = [1,2,3,4,3,2,1]
d_t_1 = 3
d_t = 3**2
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 5,
    d_sys_total = d_sys_total,
    d_t_total = [d_t_1]*2,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=32
)

help_obj = Symmetrical_Coupling_Helper(d_sys_total)
l_list = Symmetrical_Coupling_Helper.calc_l_list(taus,d_sys_total,delta_t)
help_obj.set_fback_subchain_lengths(l_list)

bond = 4
sbins=[[] for x in range(N)] 
i_s = np.zeros([1,np.prod(d_sys_total),1],dtype=complex)
#i_s[:,int(np.prod(d_sys_total)-1),:] = 1
i_s[:,3,:] = 1

nbins = qmps.simulation._separate_sys_bins(i_s, d_sys_total, sbins, bond)
c_nbins = copy.deepcopy(nbins)
for i in range(len(nbins)):
    print(i,':', c_nbins[-1-i].shape)
    print(qops.expectation_1bin(c_nbins[-1-i], qops.create(d_sys_total[i])@qops.destroy(d_sys_total[i])))
    if i == len(nbins)-1: continue
    contraction=ncon([c_nbins[-2-i],c_nbins[-1-i]],[[-1,-2,2],[2,-3,-4]])
    left_bin,stemp,c_nbins[-1-i] = _svd_tensors(contraction,bond,contraction.shape[1],contraction.shape[2])
    c_nbins[-2-i] = left_bin * stemp[None,None,:]

print('='*60)

nbins = qmps.simulation._reorder_sys_bins_sym_efficient(nbins, d_sys_total, help_obj, bond)
for i in range(len(nbins)):
    print(help_obj.ordered_indices[i], ':', nbins[-1-i].shape)
    print(qops.expectation_1bin(nbins[-1-i], qops.create(help_obj.d_sys_ordered[i])@qops.destroy(help_obj.d_sys_ordered[i])))
    if i == len(nbins)-1: continue
    contraction=ncon([nbins[-2-i],nbins[-1-i]],[[-1,-2,2],[2,-3,-4]])
    left_bin,stemp,nbins[-1-i] = _svd_tensors(contraction,bond,contraction.shape[1],contraction.shape[2])
    nbins[-2-i] = left_bin * stemp[None,None,:]

#%%% Continuation of prior example: Test the initializations of the feedback loops (filling them with field bins)
nbins_init = qmps.simulation._initialize_feedback_loop_sym_efficient(copy.deepcopy(nbins), l_list, d_t, d_sys_total, bond, help_obj, input_field_generator=None)

#%%%
print(len(nbins_init))
for i in range(len(nbins_init)):
    print(nbins_init[-1-i].shape)
    #print(nbins_init[-1-i])
    #print('='*50)
# %%
N = 8
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3,3]
delta_t = 0.05
taus = [2]*(N-1); taus = [1,2,1,2,1,2,1]; taus = [1,2,1,2,1,2,1]
d_t_1 = 3
d_t = 3**2
params = qmps.parameters.InputParams(
    delta_t = delta_t,
    tmax = 5,
    d_sys_total = d_sys_total,
    d_t_total = [d_t_1]*2,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=64
)
tmax = params.tmax
tlist=np.arange(0,tmax+params.delta_t, params.delta_t)

i_n0 = np.zeros([1,d_t,1],dtype=complex) #initial time bin

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin

# Start with first 2 in chain excited
#i_s0[:,int(2**(len(d_sys_total)-1) + 2**(len(d_sys_total)-2)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# Just First one excited
#i_s0[:,int(2**(len(d_sys_total)-1)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# All excited
i_s0[:,int(2**(len(d_sys_total))-1),:] = 1; 
i_n0 = None

gamma_ls = [0.5]*N; gamma_rs = [0.5]*N
hams = qmps.hamiltonian_Ntls_sym_eff(params, gamma_ls, gamma_rs)

#%%%
bins = qmps.t_evol_nmar_sym(hams, i_s0, i_n0, taus, params)

#%%%
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


sys_pop_op = qops.sigmaplus() @ qops.sigmaminus()
sys_pops = []
for i in range(len(d_sys_total)):
    sys_pops.append(qops.single_time_expectation(bins.system_states[i], sys_pop_op))
#out_flux = qops.single_time_expectation(out_bins, flux_op)



fonts=18
pic_style(fonts)


#fig, ax = plt.subplots(figsize=(4.5, 4))
fig, ax = plt.subplots(figsize=(7, 5))

#plt.plot(tlist,np.real(out_flux),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{R}$')
for i in range(N):
    if i < int(N/2): curr_line_style = '-'
    else: curr_line_style = '--'
    plt.plot(tlist,np.real(sys_pops[i]),linewidth = 3,linestyle=curr_line_style,label=r'$n_{\rm TLS}^{('+str(i)+r')}$')

#plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
#plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='R')
#plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.1),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,5*N])
plt.xlim([0.,5])

plt.tight_layout()
#plt.savefig('pops.pdf', bbox_inches='tight', dpi=400)

plt.show()

# %%
