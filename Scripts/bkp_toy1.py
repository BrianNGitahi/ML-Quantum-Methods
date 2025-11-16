#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import defaultdict

#%%
# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor_product(matrices):
    """Compute tensor product of a list of matrices."""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def pauli_string_to_matrix(pauli_string, n_qubits=7):
    """
    Convert a Pauli string to matrix representation.
    pauli_string: dict mapping qubit index to Pauli operator ('I', 'X', 'Y', 'Z')
    """
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    matrices = []
    for i in range(n_qubits):
        if i in pauli_string:
            matrices.append(pauli_map[pauli_string[i]])
        else:
            matrices.append(I)
    return tensor_product(matrices)

# Hamiltonian from Equation 25
hamiltonian_terms = [
    (41.75, {}),  # Identity term
    (-14.0, {0: 'Z'}),
    (-15.5, {1: 'Z'}),
    (-20.5, {2: 'Z'}),
    (-19.5, {3: 'Z'}),
    (-6.0, {4: 'Z'}),
    (-12.0, {5: 'Z'}),
    (-24.0, {6: 'Z'}),
    (7.5, {0: 'Z', 1: 'Z'}),
    (10.0, {0: 'Z', 2: 'Z'}),
    (8.75, {0: 'Z', 3: 'Z'}),
    (2.5, {0: 'Z', 4: 'Z'}),
    (5.0, {0: 'Z', 5: 'Z'}),
    (10.0, {0: 'Z', 6: 'Z'}),
    (12.0, {1: 'Z', 2: 'Z'}),
    (10.5, {1: 'Z', 3: 'Z'}),
    (3.0, {1: 'Z', 4: 'Z'}),
    (6.0, {1: 'Z', 5: 'Z'}),
    (12.0, {1: 'Z', 6: 'Z'}),
    (14.0, {2: 'Z', 3: 'Z'}),
    (4.0, {2: 'Z', 4: 'Z'}),
    (8.0, {2: 'Z', 5: 'Z'}),
    (16.0, {2: 'Z', 6: 'Z'}),
    (3.5, {3: 'Z', 4: 'Z'}),
    (7.0, {3: 'Z', 5: 'Z'}),
    (14.0, {3: 'Z', 6: 'Z'}),
    (2.0, {4: 'Z', 5: 'Z'}),
    (4.0, {4: 'Z', 6: 'Z'}),
    (8.0, {5: 'Z', 6: 'Z'}),
]

print("Building Hamiltonian matrix...")
H = np.zeros((128, 128), dtype=complex)
for coeff, pauli_string in hamiltonian_terms:
    H += coeff * pauli_string_to_matrix(pauli_string)

print("\nPerforming exact diagonalization...")
eigenvalues, eigenvectors = eigh(H)
ground_state_energy = eigenvalues[0]
ground_state = eigenvectors[:, 0]

print(f"\nGround state energy: {ground_state_energy:.6f}")
print(f"Expected energy: -12.0")
print(f"Match: {np.abs(ground_state_energy - (-12.0)) < 1e-6}")

# Find dominant basis state
probabilities = np.abs(ground_state)**2
dominant_idx = np.argmax(probabilities)
dominant_prob = probabilities[dominant_idx]

print(f"\nDominant basis state index: {dominant_idx}")
print(f"Binary representation: {format(dominant_idx, '07b')}")
print(f"Probability: {dominant_prob:.6f}")

# Verify it's essentially a pure computational basis state
if dominant_prob > 0.99:
    print("\n✓ Ground state is essentially a single basis state!")
else:
    print(f"\n⚠ Ground state has significant superposition")
    print("Top 5 basis states:")
    top_indices = np.argsort(probabilities)[-5:][::-1]
    for idx in top_indices:
        print(f"  |{format(idx, '07b')}⟩: {probabilities[idx]:.6f}")

#%%
def get_measurement_basis_for_pauli(pauli_string, n_qubits=7):
    """
    Returns which basis each qubit should be measured in.
    For diagonal Hamiltonians, all measurements are in Z basis.
    'Z' for computational basis, 'X' or 'Y' for rotated bases.
    """
    basis = ['Z'] * n_qubits
    for qubit_idx, pauli_op in pauli_string.items():
        if pauli_op == 'X':
            basis[qubit_idx] = 'X'
        elif pauli_op == 'Y':
            basis[qubit_idx] = 'Y'
        # Z and I both stay as 'Z' (computational basis measurement)
    return basis

def apply_measurement_rotation(state, basis):
    """
    Apply unitary rotation to measure in specified basis.
    Returns the transformed state.
    """
    n_qubits = int(np.log2(len(state)))
    
    # Build the rotation unitary
    rotation_matrices = []
    for b in basis:
        if b == 'Z':
            rotation_matrices.append(I)
        elif b == 'X':
            # Rotation to X basis: Hadamard
            H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            rotation_matrices.append(H_gate)
        elif b == 'Y': 
            # Rotation to Y basis
            Y_rot = np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2)
            rotation_matrices.append(Y_rot)
    
    U = tensor_product(rotation_matrices)
    return U @ state

def sample_bitstring(state_amplitudes):
    """Sample a bitstring from the quantum state probability distribution."""
    probabilities = np.abs(state_amplitudes)**2
    probabilities = probabilities / np.sum(probabilities)  # Normalize
    
    n_qubits = int(np.log2(len(state_amplitudes)))
    outcome = np.random.choice(len(state_amplitudes), p=probabilities)
    
    # Convert to bitstring
    bitstring = format(outcome, f'0{n_qubits}b')
    return bitstring

def generate_measurement_dataset(ground_state, hamiltonian_terms, n_samples):
    """
    Generate synthetic measurement dataset by sampling from the ground state
    in random Pauli bases.
    """
    n_qubits = int(np.log2(len(ground_state)))
    
    # Filter out identity term for measurement sampling
    non_identity_terms = [(coeff, ps) for coeff, ps in hamiltonian_terms if len(ps) > 0]
    
    dataset = []
    basis_counts = defaultdict(int)
    
    print(f"\nGenerating {n_samples} measurement samples...")
    for i in range(n_samples):
        # Randomly select a Pauli term (uniformly)
        _, pauli_string = non_identity_terms[np.random.randint(len(non_identity_terms))]
        
        # Get measurement basis
        basis = get_measurement_basis_for_pauli(pauli_string, n_qubits)
        basis_label = ''.join(basis)
        basis_counts[basis_label] += 1
        
        # Apply rotation and sample
        rotated_state = apply_measurement_rotation(ground_state, basis)
        bitstring = sample_bitstring(rotated_state)
        
        dataset.append({
            'bitstring': bitstring,
            'basis': basis_label,
            'pauli_term': pauli_string
        })
        
        if (i + 1) % (n_samples // 10) == 0:
            print(f"  Progress: {i + 1}/{n_samples}")
    
    print(f"\n✓ Generated {len(dataset)} measurements")
    print(f"  Unique measurement bases used: {len(basis_counts)}")
    
    return dataset, basis_counts

#%%
# Generate datasets of different sizes
dataset_sizes = [1000, 10000, 100000]

for M in dataset_sizes:
    print(f"\n{'='*60}")
    print(f"Generating dataset with M = {M:,} samples")
    print('='*60)
    
    dataset, basis_counts = generate_measurement_dataset(
        ground_state, 
        hamiltonian_terms, 
        M
    )
    
    # Analyze the dataset
    print("\nDataset statistics:")
    bitstring_counts = defaultdict(int)
    for sample in dataset:
        bitstring_counts[sample['bitstring']] += 1
    
    print(f"  Unique bitstrings observed: {len(bitstring_counts)}")
    print(f"  Most common bitstring: {max(bitstring_counts, key=bitstring_counts.get)}")
    print(f"    (appears {bitstring_counts[max(bitstring_counts, key=bitstring_counts.get)]} times)")
    
    # Save dataset
    filename = f'synthetic_measurements_M{M}.npz'
    np.savez(filename,
             bitstrings=np.array([s['bitstring'] for s in dataset]),
             bases=np.array([s['basis'] for s in dataset]),
             ground_state=ground_state,
             hamiltonian_terms=hamiltonian_terms,
             ground_state_energy=ground_state_energy,
             n_qubits=7)
    print(f"\n✓ Saved dataset to {filename}")

print("\n" + "="*60)
print("Dataset generation complete!")
print("="*60)
# %%
