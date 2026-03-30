import numpy as np
# We import your class from the moha folder
from moha.hamiltonians import AlternativeSpinHamiltonian

# Let's test with 2 spins (Matrix should be 4x4)
n_spins = 2
couplings = [1.0, 1.0] 

print(f"--- Testing AlternativeSpinHamiltonian with {n_spins} spins ---")
model = AlternativeSpinHamiltonian(n_spins, couplings)

# Get Sz for the first spin (index 0)
sz0 = model.get_sz_operator(0)

print("Matrix for Sz(0):")
print(sz0)
print(f"Shape: {sz0.shape}")

# Verification: The trace of Sz should be 0
if np.trace(sz0) == 0:
    print("\n✅ Success: Trace is 0. The physics looks correct.")
else:
    print("\n❌ Error: Trace is not 0. Check the Sz matrix definition.")
    
from moha.hamiltonians import AlternativeSpinHamiltonian
import numpy as np

# 2 spins, connected (0, 1)
conn = [(0, 1)]
model = AlternativeSpinHamiltonian(2, [1.0], connectivity=conn)

H = model.generate_integrals()
print("Full Hamiltonian Matrix (Heisenberg 2-spin):")
print(H)

# Physics check: Eigenvalues (Energy levels)
energies = np.linalg.eigvals(H)
print("\nEnergy Levels (Eigenvalues):")
print(np.sort(energies.real))    