# Usage

This section introduces the basic structure of QwaveMPS and its functionalities. The following diagram summarizes the QwaveMPS framework,


![Diagram of the code workflow](./images/diagram.png)

## Modules

These are the main modules and scripts in this repository:

- `parameters.py`: This module contains the InputParams and Bins classes used throughout the library
- `hamiltonians.py`: This module contains the Hamiltonians for different cases
- `operators.py`: This module contains the main quantum operators, including the basic boson and TLS operators used to build the Hamiltonian, and observable operators. It also contains single time expectation functions. 
- `simulation.py`: This module contains the functions to evolve the system, both in the Markovian and non-Markovian regimes.
- `states.py`: This module contains initial states for the waveguide and the TLSs.
- `correlation.py`: This module contains functions to calculate two time correlation functions of the output field.

## Input Parameters
The diagram above contains the basic components of the package. The input parameters include the initial parameters needed to set up the system and run the main simulation (parameters.py), the initial states of the emitter and the field (states.py), and the Hamiltonian of the system (hamiltonian.py). The time evolution of the total system state is calculated (stored in the Bins data class) using a Markovian or non-Markovian time evolution function (simulation.py). 

## Output
Observables such as TLS populations and field fluxes can be calculated by taking the expectation values of the relevant operators on the appropriate emitter/field bins (operators.py). In a similar way, two time correlation functions can also be calculated (correlation.py). An example output plot with some observables in the non-Markovian regime is included in the bottom right of the above diagram.
