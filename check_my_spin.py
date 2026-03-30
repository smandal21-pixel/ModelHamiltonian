import numpy as np
from moha.hamiltonians import AlternativeSpinHamiltonian

# Initialize 2-spin system with homogeneous coupling
n_spins = 2
couplings = [1.0] 
model = AlternativeSpinHamiltonian(n_spins, couplings)

print(f"Testing AlternativeSpinHamiltonian (Spins: {n_spins})")
H_matrix = model.generate_integrals()

print("\nHamiltonian Matrix:")
print(H_matrix)

# Verify eigenvalues against analytical solutions for 2-spin Heisenberg model
energies = np.linalg.eigvals(H_matrix)
sorted_energies = np.sort(energies.real)

print("\nCalculated Eigenvalues:")
print(sorted_energies)

expected = np.array([-0.75, 0.25, 0.25, 0.25])
if np.allclose(sorted_energies, expected):
    print("\nStatus: Passed ")
else:
    print("\nStatus: Failed - Eigenvalues deviate from analytical expectations.")