import numpy as np
import pytest
from moha.hamiltonians import AlternativeSpinHamiltonian

def test_heisenberg_2spin_energies():
    """Verify the 2-spin Heisenberg model produces exact analytical eigenvalues."""
    n_spins = 2
    couplings = [1.0]
    model = AlternativeSpinHamiltonian(n_spins, couplings)

    H = model.generate_integrals()
    energies = np.linalg.eigvals(H)
    sorted_energies = np.sort(energies.real)

    # Expected eigenvalues for J=1: [-0.75, 0.25, 0.25, 0.25]
    expected = np.array([-0.75, 0.25, 0.25, 0.25])
    assert np.allclose(sorted_energies, expected), f"Expected {expected}, got {sorted_energies}"

def test_hermiticity():
    """The Hamiltonian must be Hermitian (H = H.T)."""
    model = AlternativeSpinHamiltonian(3, [1.0, 1.0])
    H = model.generate_integrals()
    assert np.allclose(H, H.T), "Hamiltonian is not Hermitian!"