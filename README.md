# QwaveMPS
> QwaveMPS is an open-source Python package that calculates light-matter interactions in waveguide QED systems using Matrix Product States, allowing one to study delayed feedback effects in the non-Markovian regime.

QwaveMPS is an open-source Python library for simulating one-dimensional quantum many-body systems using Matrix Product States (MPS). Designed for researchers and students, it provides a user-friendly interface for constructing, evolving, and analyzing quantum states and operators, facilitating studies in quantum physics and quantum information. This approach enables efficient, scalable simulations by focusing computational resources on the most relevant parts of the quantum system. Thus, one can study delayed feedback effects in the non-Markovian regime at a highly reduced computational cost compared to full Hilbert space approaches, making it practical to model open waveguide QED systems.


## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citing](#citing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

This package can be installed using the following command:
```
pip install QwaveMPS
```
<!--For now I am cloning the repository, going to the root folder, and installing it with 
```
pip install .
```
To be able to use the first option, we need to first publish it in PyPI. -->


## Features

- **Nonlinear non-Markovian waveguide-QED problems:** Solve problems involving multiphoton and multiatom systems with time-delayed feedback.
- **Matrix product states (MPS):** Based on MPS theory for efficient numerical exact results that do not rely on Monte Carlo simulations.
- **Open-source Python package:** Python library with minimal dependencies on external packages.
- **User-friendly framework:** Designed to be accessible for researchers and students.
- **Comprehensive Resources:** Documentation and examples provided to support learning and usage.
  
## Usage

The following diagram summarizes the Qwavwmps framework, showing the main input and output parameters. For detailed usage information, see our [documentation](docs/usage.md).

![Diagram of the code workflow](docs/images/diagram.png)

## Contributing
Check our [contributing guidelines](docs/contributing.md) for details on how to contribute to QwaveMPS.

## Citing

Once we have the paper/arxiv, add here how to cite the repo

## License

QwaveMPS is distributed under the GNU GPLv3. See [LICENSE](LICENSE) for more information.

## Acknowledgements

Add acknowledgements here.

## Contact
For questions or support, open an issue or email [18sar4@queensu.ca](mailto:18sar4@queensu.ca).
