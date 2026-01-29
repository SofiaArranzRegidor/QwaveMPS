# Modules

These are the main modules and scripts in this repository:

- `hamiltonians.py`: This module contains the Hamiltonians for different cases
- `operators.py`: This module contains the main quantum operators, including the basic boson and TLS operators used to build the Hamiltonian, the time evolution operator, swap operator, and expectation and observable operators. 
- `simulation.py`: This module contains the functions to apply SVDs, evolve the system both in the Markovian and non-Markovian regimes, and functions to calculate the main observables.
- `states.py`: This module contains initial states for the waveguide and the TLSs.
- `correlation.py`: This module contains functions to calculate two time point correlations.
- `parameters.py`: This module contains InputParams data class used in the setup of the simulation, and the Bins data class that contains the results of the simulation.

