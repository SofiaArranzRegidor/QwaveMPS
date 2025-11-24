# Theory

## Waveguide QED

Waveguide quantum electrodynamics (QED) systems study quasi one-dimensional quantum systems where atoms or quantum emitters couple to a continuum of quantized field modes. These systems show new phenomena
that are unique to the waveguide geometry, such as significant non-Markovian or delayed feedback effects, which can lead to the emission and reabsorption of photons by the quantum emitters, or the ability
to couple to chiral emitters, breaking the local symmetry of the problem by emitting photons only in one waveguide direction for a better flow of the quantum information.

However, to solve these complex problems, many theories make use of approximations, such as the Markovian approximation, which results in valuable information being lost. In this project, we give a tool to solve waveguide QED problems using matrix product states (MPS), which allows us to solve these systems without making some of the usual approximations.  

## Matrix Product States
### Singular value decomposition

Matrix product states is an approach based on one-dimensional tensor network theories (add citation). The MPS algorithm relies on the Schmidt or singular value decomposition (SVD) of a quantum system, which considers the bipartition state of the system as a tensor product. The SVD decomposition of a tensor $M$ is,

$$M = U S V^\dagger,$$

where $S$ is a diagonal matrix containing the Schmidt coefficients in descending order, $U$ is a left-normalized tensor, and $V$ is a right-normalized one. Afterwards, one of the side tensors can be contracted with the tensor containing the Schmidt coefficients. This receives the name of the orthogonality center (OC) and carries the system's information. Thus, we end up with 2 new tensors written as a tensor product. To better understand the process, this can be represented diagrammatically,

<p align="center">
  <img src="/images/OC.png" alt="OC Image" width="30%">
</p>

### Matrix product states

By iterating this process, we can decompose the Hilbert space into a tensor product of smaller subspaces until we get the following general MPS expression for a waveguide QED system,

$$\ket{\psi}=\sum_{i_s i_1...i_N} A_{a_1}^{i_s}A_{a_1,a_2}^{i_1} ... A_{a_{N-1},a_{N}}^{i_{N-1}}A_{a_{N}}^{i_{N}}\ket{i_s, i_1,...,i_N},$$

where the first term represents the system (or quantum emitter) part, and the remaining $N$ terms represent the waveguide discretized in time. Here, each tensor can be represented as a 'bin' which corresponds to the boxes in the diagrammatic representation. This gives the possibility of at least $N$ photons in the waveguide.

<p align="center">
  <img src="./images/initial_state.png" alt="state Image" width="50%">
</p>

For example, if our system contains a single TLS that starts excited, the TLS bin is represented by,  

```sh
def i_se(d_sys1:int=2, bond0:int=1) -> np.ndarray:
    i_s = np.zeros([bond0,d_sys1,bond0],dtype=complex) 
    i_s[:,1,:]=1.
    return i_s
```
And if the waveguide field starts in vacuum, each time bin will follow,

```sh
def i_ng(d_t:int, bond0:int=1) -> np.ndarray:
    i= np.zeros([bond0,d_t,bond0],dtype=complex) 
    i[:,0,:]=1.
    return i
```
with the total field being a tensor product of these time bins.

### Matrix product operators

<p align="center">
  <img src="./images/mpo.png" alt="mpo Image" width="15%">
</p>

## References

List important papers or resources for readers interested in the theory.
