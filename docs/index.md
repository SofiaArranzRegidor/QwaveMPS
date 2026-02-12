
```{image} _static/logo.svg
:alt: QwaveMPS logo
:width: 180px
:align: center
```

# QwaveMPS 


> QwaveMPS is an open-source Python package that calculates light-matter interactions in waveguide QED systems using Matrix Product States, allowing one to study delayed feedback effects in the non-Markovian regime.


QwaveMPS is an open-source Python library for simulating one-dimensional quantum many-body systems using Matrix Product States (MPS). 
Designed for researchers and students, it provides a user-friendly interface for constructing, evolving, and analyzing quantum states and operators, 
facilitating studies in quantum physics and quantum information. This approach enables efficient, scalable simulations 
by focusing computational resources on the most relevant parts of the quantum system. Thus, one can study delayed feedback effects in the non-Markovian 
regime at a highly reduced computational cost compared to full Hilbert space approaches, making it practical to model open waveguide QED systems.



```{toctree}
:maxdepth: 2
:hidden:
installation
theory
usage
auto_examples/index
api-reference/index
references
```


## Features


::::{grid} 1 2 2 3
:gutter: 1

:::{grid-item-card} Nonlinear non-Markovian waveguide-QED problems
Solve problems involving multiphoton and multiatom systems with time-delayed feedback.
:::

:::{grid-item-card} Matrix product states (MPS)
Based on MPS theory for efficient numerical exact results that do not rely on Monte Carlo simulations.
:::

:::{grid-item-card} Open-source Python package
Python library with minimal dependencies on external packages.
:::

:::{grid-item-card} User-friendly framework
Designed to be accessible for researchers and students.
:::

:::{grid-item-card} Comprehensive Resources
Documentation and examples provided to support learning and usage.
:::

:::: 



## Download and installation

The source repository is [available for download](https://github.com/SofiaArranzRegidor/QwaveMPS) on GitHub. The installation instructions are given [here](installation.md). 

## Usage summary

The following diagram summarizes the QwaveMPS framework, showing the main input and output parameters. For detailed usage information, 
see the [Usage](usage.md) and [Examples](auto_examples/index) sections.

![Diagram of the code workflow](./images/diagram.png)


## License

QwaveMPS is distributed under the GNU GPLv3. See [LICENSE](../LICENSE) for more information.

## Contact
For questions or support, open an issue or email [qwavemps@gmail.com](mailto:qwavemps@gmail.com).

If you encounter a bug or have a feature request, please open an issue on the [issue tracker](https://github.com/SofiaArranzRegidor/QwaveMPS/issues) to report it. 

