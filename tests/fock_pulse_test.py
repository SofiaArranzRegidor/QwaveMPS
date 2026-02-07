#%% 
# Imports
#--------

import QwaveMPS as qmps
import matplotlib.pyplot as plt
import numpy as np
import time as t


#%% 

""""Choose the simulation parameters"""
N = 6
d_t_l=N+1 #Time right channel bin dimension
d_t_r=N+1 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 6,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=16
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state and tophat pulse parameters"""
sys_initial_state=qmps.states.tls_ground()


# Pulse parameters for a 2-photon gaussian pulse
photon_num = N #number of photons
gaussian_center = 3
gaussian_width = 1

#pulse_time = tmax
#pulse_envelope = qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)

pulse_time = 1
pulse_envelope = None

wg_initial_state = qmps.states.fock_pulse(pulse_envelope,pulse_time, photon_num, input_params, direction='R')

start_time=t.time()

#Calculate the time evolution
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""Calculate time evolution of the system"""
# Create the Hamiltonian again for this larger Hilbert space
Hm=np.zeros((np.kron(qmps.b_pop_l(input_params), qmps.tls_pop())).shape)
# Hm = qmps.hamiltonian_1tls(input_params)

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


#Calculate the population dynamics
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""Calculate population dynamics"""
photon_flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

# Calculate same time G2 in transmission
same_time_corrs = [qmps.b_pop_r(input_params)]
for i in range(N+1):
    same_time_corrs.append(qmps.b_dag_r(input_params) @ same_time_corrs[-1] @ qmps.b_r(input_params))

tls_pop = qmps.single_time_expectation(bins.system_states, qmps.tls_pop())
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_flux_ops)
same_time_corrs = qmps.single_time_expectation(bins.output_field_states, same_time_corrs)

# Act on input states to characterize the input field with bosonic/field operators
flux_in = qmps.single_time_expectation(bins.input_field_states, photon_flux_ops)


total_quanta = tls_pop + np.cumsum(photon_fluxes[0] + photon_fluxes[1]) * delta_t

#^^^^^^^^^^^^^^^^
#Plot the results
#^^^^^^^^^^^^^^^^

#plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'violet',linestyle='-',label=r'$n_{R}$') # Photons transmitted to the right channel
#plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'green',linestyle=':',label=r'$n_{L}$') # Photons reflected to the left channel
#plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
#plt.plot(tlist,np.real(flux_in[1]),linewidth = 3, color = 'grey',linestyle='--',label=r'$n_{R}^{\rm in}$') # Photon flux in from right
plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
line_styles=['-', '--', ':']
for i in range(N+1):
    plt.plot(tlist, np.real(same_time_corrs[i]), linewidth=3, linestyle=line_styles[i%3], label=r'$G_{RR}^{(%s)}$' % (i+1))


plt.legend()
plt.xlabel(r'Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,None])
plt.xlim([0.,tmax])
plt.show()

print('Correlation \t Value \t\t Integration')
for i in range(N+1):
    index= np.argmax(np.real(same_time_corrs[i]))
    print(f"{i+1} \t\t {np.real(same_time_corrs[i][index]):.8f} \t {np.real(np.sum(same_time_corrs[i]*delta_t)):.8f}")

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Calculate the two-time correlations 
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#Here, we show how to calculate the second-order correlation
#which will have values different from 0 since the pulse 
#contains now 2 photons.
#

#To track computational time of G2
start_time=t.time()

# For speed calculating several at once, but could also calculate all at once
a_op_list = []; b_op_list = []; c_op_list = []; d_op_list = []

# Have to create operators again for this larger space
b_dag_l = qmps.b_dag_l(input_params); b_l = qmps.b_l(input_params)
b_dag_r = qmps.b_dag_r(input_params); b_r = qmps.b_r(input_params)

# Add op <b_R^\dag(t) b_R^\dag(t+t') b_R^(t+t') b_R(t)>
a_op_list.append(b_dag_r)
b_op_list.append(b_dag_r)
c_op_list.append(b_r)
d_op_list.append(b_r)

# Add op <b_L^\dag(t) b_L^\dag(t+t') b_L^(t+t') b_L(t)>
a_op_list.append(b_dag_l)
b_op_list.append(b_dag_l)
c_op_list.append(b_l)
d_op_list.append(b_l)


# Add op <b_R^\dag(t) b_L^\dag(t+t') b_L^(t+t') b_R(t)>
a_op_list.append(b_dag_r)
b_op_list.append(b_dag_l)
c_op_list.append(b_l)
d_op_list.append(b_r)

# Add op <b_L^\dag(t) b_R^\dag(t+t') b_R^(t+t') b_L(t)>
a_op_list.append(b_dag_l)
b_op_list.append(b_dag_r)
c_op_list.append(b_r)
d_op_list.append(b_l)

# Could also consider G1 correlation functions in the same call if we were interested
# For example: <b_R^\dag(t)b_R(t+t')> 
a_op_list.append(b_dag_r)
b_op_list.append(b_r)
c_op_list.append(np.eye(input_params.d_t))
d_op_list.append(np.eye(input_params.d_t))

g2_correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins, a_op_list, b_op_list, c_op_list, d_op_list, input_params)


print("G2 correl--- %s seconds ---" %(t.time() - start_time))

#%%
#^^^^^^^^^^^^^^^^
#Plot the results
#^^^^^^^^^^^^^^^^
#

X,Y = np.meshgrid(correlation_tlist,correlation_tlist)


"""Example graphing G2_{RR}"""
# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(qmps.transform_t_tau_to_t1_t2(g2_correlations[0]))
absMax = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap='Reds', vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax)
ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
cbar.set_label(r'$G^{(2)}_{RR}(t,t^\prime)\ [\gamma^{2}]$')
plt.show()


"""Example graphing G2_{LL}"""
z = np.real(qmps.transform_t_tau_to_t1_t2(g2_correlations[1]))
absMax = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap='Reds', vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax)
ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
cbar.set_label(r'$G^{(2)}_{LL}(t,t^\prime)\ [\gamma^{2}]$')
plt.show()

"""Example graphing G2_{LR}"""
# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
# Since the correlation isn't symmetric w.r.t. t', need both G2_{LR} and G2_{RL}
# Arguments below would be reversed for G2_{RL}
z = np.real(qmps.transform_t_tau_to_t1_t2(g2_correlations[3],g2_correlations[2]))
absMax = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap='Reds', vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax)
ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
cbar.set_label(r'$G^{(2)}_{LR}(t,t^\prime)\ [\gamma^{2}]$')
plt.show()
