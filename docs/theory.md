# Theory

## Waveguide QED

Waveguide quantum electrodynamics (QED) systems study quasi one-dimensional quantum systems where atoms or quantum emitters couple to a continuum of quantized field modes. These systems show new phenomena
that are unique to the waveguide geometry, such as significant non-Markovian or delayed feedback effects, which can lead to the emission and reabsorption of photons by the quantum emitters, or the ability
to couple to chiral emitters, breaking the local symmetry of the problem by emitting photons only in one waveguide direction for a better flow of the quantum information.

However, to solve these complex problems, many theories make use of approximations, such as the Markovian approximation, which results in valuable information being lost. In this project, we give a tool to solve waveguide QED problems using matrix product states (MPS), which allows us to solve these systems without making some of the usual approximations.  

## Matrix Product States

The quantum state for a 1D spin-chain, with $N$ spins, is given by \cite{woolfe_matrix_2015},
\begin{equation}
    \ket{\psi}= \sum_{i_1,...,i_N}^{d} c_{i_1,...,i_N} \ket{i_1,...,i_N},
    \label{eq:psi}
\end{equation}
%
where $i_k$ (with $k \ \in \ \{1,...N\}$) represents each state with a dimension of $d$, and $c_i$ are the coefficients of the corresponding state. 

The MPS algorithm relies on the Schmidt decomposition of a quantum system, which considers  
the bipartition state of the system 
as 
 a tensor product

## References

List important papers or resources for readers interested in the theory.
