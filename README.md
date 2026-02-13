<p align="center">
  <img src="https://raw.githubusercontent.com/SofiaArranzRegidor/QwaveMPS/main/docs/_static/logo.svg" alt="QwaveMPS logo" width="180">
</p>

<h1 align="center">QwaveMPS</h1>

<div align="center">

[![build](https://github.com/SofiaArranzRegidor/QwaveMPS/actions/workflows/build_main.yaml/badge.svg)](https://github.com/SofiaArranzRegidor/QwaveMPS/actions/workflows/build_main.yaml)
[![Codecov](https://img.shields.io/codecov/c/github/SofiaArranzRegidor/QwaveMPS?token=52MBM273IF)](https://codecov.io/gh/SofiaArranzRegidor/QwaveMPS)
[![Documentation Status](https://readthedocs.org/projects/QwaveMPS/badge/?version=latest)](https://pycharge.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/QWaveMPS.svg)](https://pypi.org/project/QWaveMPS/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/SofiaArranzRegidor/QwaveMPS?color=blue)](https://github.com/SofiaArranzRegidor/QwaveMPS/blob/main/LICENSE)

</div>

<p align="center">
  Matrix-product-state simulations of non-Markovian waveguide QED
</p>

> QwaveMPS is an open-source Python package that calculates light-matter interactions in waveguide QED systems using Matrix Product States, allowing one to study delayed feedback effects in the non-Markovian regime.

QwaveMPS is an open-source Python library for simulating one-dimensional quantum many-body systems using Matrix Product States (MPS). Designed for researchers and students, it provides a user-friendly interface for constructing, evolving, and analyzing quantum states and operators, facilitating studies in quantum physics and quantum information. This approach enables efficient, scalable simulations by focusing computational resources on the most relevant parts of the quantum system. Thus, one can study delayed feedback effects in the non-Markovian regime at a highly reduced computational cost compared to full Hilbert space approaches, making it practical to model open waveguide QED systems.

## Features

- **Nonlinear non-Markovian waveguide-QED problems:** Solve problems involving multiphoton and multiatom systems with time-delayed feedback.
- **Matrix product states (MPS):** Based on MPS theory for efficient numerical exact results that do not rely on Monte Carlo simulations.
- **Open-source Python package:** Python library with minimal dependencies on external packages.
- **User-friendly framework:** Designed to be accessible for researchers and students.
- **Comprehensive Resources:** Documentation and examples provided to support learning and usage.

## Installation

This package can be installed using the following command:
```
pip install QwaveMPS
```
  
## Usage

The following diagram summarizes the QwaveMPS framework, showing the main input and output parameters. For detailed usage information, see our [documentation](docs/usage.md).

![Diagram of the code workflow](docs/images/diagram.png)

## Documentation

Read the full documentation at [https://qwavemps.readthedocs.io](https://qwavemps.readthedocs.io)

## Simple example: Population dynamics of a TLS in an infinite waveguide

Import the necessary packages:
```python
import numpy as np
import QwaveMPS as qmps
```
Define the simulation parameters:
```python
delta_t = 0.05 # Time step of the simulation
tmax = 8 # Maximum simulation time
tlist=np.arange(0,tmax+delta_t,delta_t)
d_t_l=2 #Size of right-channel time bin (choose 2 for 1 photon per bin)
d_t_r=2 #Size of left-channel time bin 
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin

gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, 
    tmax = 8,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r=gamma_r,  
    bond_max=4 # Maximum bond dimension, truncates entanglement information
)
```
Choose the initial state:
```python
sys_initial_state=qmps.states.tls_excited() #TLS initially excited
wg_initial_state = qmps.states.vacuum(tmax,input_params) #waveguide in vacuum
```
Choose the Hamiltonian:
```python
ham = qmps.hamiltonian_1tls(input_params)
```
Calculate time evolution of the system:
```python
bins = qmps.t_evol_mar(hm,sys_initial_state,wg_initial_state,input_params)
```
Choose operators to calculate population dynamics:
```python
tls_pop_op = qmps.tls_pop()
flux_op_l = qmps.b_pop_l(input_params)
flux_op_r = qmps.b_pop_r(input_params)
flux_ops = [flux_op_l, flux_op_r]
```
Calculate population dynamics with operators or lists of operators:
```python
tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, flux_ops)
```
Plot of population dynamics:


<p align="center">
  <img src="docs/images/TLS_sym_decay.png" alt="state Image" width="40%">
</p>

Check [the full example](https://qwavemps.readthedocs.io/en/latest/auto_examples/Example_1TLS_M.html) for more details 


<!--## Citing

Once we have the paper/arxiv, add here how to cite the repo-->

## License

QwaveMPS is distributed under the GNU GPLv3. See [LICENSE](LICENSE) for more information.

<!--## Acknowledgements

Add acknowledgements here.-->

## Contact
For questions or support, open an issue or email [qwavemps@gmail.com](mailto:qwavemps@gmail.com).

If you encounter a bug or have a feature request, please open an issue on the [issue tracker](https://github.com/SofiaArranzRegidor/QwaveMPS/issues) to report it. 
