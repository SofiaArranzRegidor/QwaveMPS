# Usage

This section introduces the basic structure of QwaveMPS and its functionalities. 


![Diagram of the code workflow](./images/diagram.png)

## Input Parameters
The diagram above contains the basic components of the package. The input parameters include the initial parameters needed to set up the system and run the main simulation (parameters.py), the initial states of the emitter and the field (states.py), and the Hamiltonian of the system (hamiltonian.py). The time evolution of the total system state is calculated (stored in the Bins data class) using a Markovian or non-Markovian time evolution function (simulation.py). 

## Output
Observables such as TLS populations and field fluxes can be calculated by taking the expectation values of the relevant operators on the appropriate emitter/field bins. In a similar way two time correlation functions can also be calculated (correlation.py).An example output plot with some observables in the non-Markovian regime is included in the bottom right.
