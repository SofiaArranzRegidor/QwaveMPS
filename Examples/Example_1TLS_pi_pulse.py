"""
1 TLS - Drive with Classical pi-pulse in semi-infinite waveguide
================================================================

This is an example of a single two-level system (TLS) in a waveguide with a side mirror
driven by a classical Rabi field pi pulse from above the waveguide. 

All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

Computes time evolution, population dynamics, steady-state correlations,
and the emission spectrum, with the following example plots:
    
1. TLS population dynamics

2. First and second-order full correlations at two time points
        
"""
#%% 
# Imports
#--------

import QwaveMPS as qmps
import matplotlib.pyplot as plt
import numpy as np
import time as t

#%%
#Population dynamics
#----------------------------------
#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose the simulation parameters
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

""""Choose the simulation parameters"""
#Choose the bins:
# Dimension chosen to be 2 to as TLS only results in emission in single quanta subspace per unit time
d_t=3 #Time channel bin dimension
d_t_total=np.array([d_t])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.02,
    tmax = 10,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=18,
    tau=1,
    phase=np.pi
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+(delta_t/2),delta_t)


#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose the initial state and Hamiltonian
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#In this case, we need to also specify the pulse information
#that will go in the Hamiltonian as an additional input


""" Choose the initial state"""
sys_initial_state=qmps.states.tls_ground()
wg_initial_state = None


"""Choose the Hamiltonian"""
#Pi pulse from above
pulse_time = tmax
gaussian_center = 1.5
gaussian_width = 0.5
pulsed_pump = np.pi * qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)

# Non-Markov Hamiltonian of a 1TLS pumped (from above) by a pi pulse
Hm=qmps.hamiltonians.hamiltonian_1tls_feedback(input_params,pulsed_pump)

#To track computational time
start_time=t.time()

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Calculate the time evolution
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#Time evolution calculation in the non-Markovian regime


"""Calculate time evolution of the system"""
bins = qmps.simulation.t_evol_nmar(Hm,sys_initial_state,wg_initial_state,input_params)

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose and calculate the observables
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

"""Define relevant photonic operators"""
flux_op= qmps.b_pop(input_params)

"""Calculate population dynamics"""
tls_pop = qmps.single_time_expectation(bins.system_states,qmps.tls_pop())

# Calculate the flux out of the system (exiting the loop)
transmitted_flux = qmps.single_time_expectation(bins.output_field_states, flux_op)

# If we want to calculate the net transmitted quanta have to integrate the flux
net_transmitted_quanta = np.cumsum(transmitted_flux) * delta_t

# If we want to calculate the net transmitted quanta have to integrate the flux
net_transmitted_quanta = np.cumsum(transmitted_flux) * delta_t

# Calculate the flux into the feedback loop
loop_flux = qmps.single_time_expectation(bins.loop_field_states, flux_op)
#  in the feedback loop
loop_sum = qmps.loop_integrated_statistics(loop_flux, input_params)

print("--- %s seconds ---" %(t.time() - start_time))

#%%
#^^^^^^^^^^^^^^^^
#Plot the results
#^^^^^^^^^^^^^^^^
#

plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(net_transmitted_quanta),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}$') # Photon flux transmitted to the right channel
plt.plot(tlist,np.real(loop_sum),linewidth = 3,color = 'b',linestyle=':',label=r'$N^{\rm loop}$') # Photon flux transmitted to the left channel
plt.legend()
plt.xlabel(r'Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([0.,10])
plt.show()

#%%
#Two-time correlations
#----------------------------------
#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose the observables for the correlations
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

"""Calculate two time correlation"""
start_time=t.time()

# Choose operators of which to correlations
a_op_list = []; b_op_list = []; c_op_list = []; d_op_list = []
b_dag = qmps.b_dag(input_params); b = qmps.b(input_params)
dim = b.shape[0]

# Add op <b_R^\dag(t) b_R(t+t')>
a_op_list.append(b_dag)
b_op_list.append(b)
c_op_list.append(np.eye(dim))
d_op_list.append(np.eye(dim))


# Add op <b_R^\dag(t) b_R^\dag(t+tau) b_R(t+tau) b_R(t)>
a_op_list.append(b_dag)
b_op_list.append(b_dag)
c_op_list.append(b)
d_op_list.append(b)

# Calculate the correlation
correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins,
                                    a_op_list, b_op_list, c_op_list, d_op_list, input_params)

print("Correlation time --- %s seconds ---" %(t.time() - start_time))

#%%
#^^^^^^^^^^^^^^^^
#Plot the results
#^^^^^^^^^^^^^^^^
#

"""Example graphing G1_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)

# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(qmps.transform_t_tau_to_t1_t2(correlations[0]))
abs_max = np.abs(z).max()


fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap='Reds', vmin=0, vmax=abs_max,rasterized=True)
cbar = fig.colorbar(cf,ax=ax)
ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma t^\prime$')
ax.set_xlim([0,6])
ax.set_ylim([0,6])
cbar.set_label(r'$G^{(1)}_{RR}\ [\gamma]$')
plt.show()

"""Example graphing G1_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)

# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(qmps.transform_t_tau_to_t1_t2(correlations[1]))
abs_max = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap='Reds', vmin=0, vmax=abs_max,rasterized=True)
cbar = fig.colorbar(cf,ax=ax)
ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
ax.set_xlim([0,6])
ax.set_ylim([0,6])
cbar.set_label(r'$G^{(2)}_{RR}\ [\gamma^{2}]$')
plt.show()
