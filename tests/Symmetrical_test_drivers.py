#%% Imports
import QwaveMPS as qmps
import QwaveMPS.operators as qops
from QwaveMPS.symmetrical_coupling_helper import Symmetrical_Coupling_Helper
from QwaveMPS.simulation import _svd_tensors
import numpy as np
import scipy as sci
from ncon import ncon
import copy
#%% Test the separation of the system bin, and it's reordering in preparation of the MPS
N = 5
d_sys_total = [2]*N; #d_sys_total = [3,2,4,3,4,2,3]
delta_t = 1
taus = [2]*(N-1)

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

#%% Continuation of prior example: Test the initializations of the feedback loops (filling them with field bins)
