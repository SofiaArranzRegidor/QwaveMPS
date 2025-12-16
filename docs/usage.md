# Usage

This section introduces the basic structure of QwaveMPS and its functionalities. 


![Diagram of the code workflow](./images/diagram.png)

## Input Parameters

The diagram below contains the basic components of the package. There are four source scripts that are part of our input parameters. Two of them contain information for the initial parameters: these are the possible initial states (states.py) and Hamiltonians (hamiltonians.py), and they are used along with the other initial parameters to set up the particular conditions for each case (left column on the diagram). The operators.py script is mostly implicitly used and contains the basic quantum operators used in all the scripts (represented with a gray box in the diagram). The last script (simulation.py) contains the main simulation components and makes use of the initial parameters to calculate the output ones (middle column in the diagram).

## Output
The right column of the diagram shows some of the possible outputs that the user can calculate with the previous inputs. As we will see in the next sections, this includes population dynamics, but is not limited to that, and we can also calculate correlations, entanglement and spectra, among others.

