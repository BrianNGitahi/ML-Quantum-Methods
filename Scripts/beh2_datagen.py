#%%
import numpy as np
import netket as nk
from scipy.linalg import eigh
from collections import defaultdict, Counter
from functools import reduce

# Define Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

#%%
def tensor_product(matrices):
    """Compute tensor product of a list of matrices."""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def apply_measurement_rotation(state, basis_string):
    """
    Apply unitary rotation to measure in specified basis.
    basis_string: string like 'XIYIZI' where each char is I, X, Y, or Z
    """
    n_qubits = int(np.log2(len(state)))
    
    # Build rotation matrices
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
    
    rotation_matrices = []
    for base in basis_string:
        if base == 'Z' or base == 'I':
            rotation_matrices.append(I)
        elif base == 'X':
            rotation_matrices.append(H_gate)
        elif base == 'Y':
            rotation_matrices.append(S_dag @ H_gate)
        else:
            raise ValueError(f"Unknown basis character: {base}")
    
    U = tensor_product(rotation_matrices)
    return U @ state

def sample_bitstring(state_amplitudes):
    """Sample a bitstring from quantum state probability distribution."""
    probabilities = np.abs(state_amplitudes)**2
    probabilities = probabilities / np.sum(probabilities)
    
    n_qubits = int(np.log2(len(state_amplitudes)))
    outcome = np.random.choice(len(state_amplitudes), p=probabilities)
    
    bitstring = format(outcome, f'0{n_qubits}b')
    return bitstring

def OperatorFromString(op_string):
    """Convert Pauli string to matrix and sites."""
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    OpList = []
    Sites = []
    
    for k, char in enumerate(op_string):
        if char in pauli_map:
            OpList.append(pauli_map[char])
            Sites.append(k)
        else:
            raise ValueError(f"Unknown Pauli character: {char}")
    
    return Sites, reduce(np.kron, OpList)

def MeasurementRotationFromString(basis_string):
    """
    Convert basis string to rotation operator matrix for measurements.
    
    For measurement bases, we need rotation unitaries, not Pauli operators:
    - I or Z: Identity (no rotation, measure in computational basis)
    - X: Hadamard (rotation to X eigenbasis)
    - Y: S†H (rotation to Y eigenbasis)
    """
    
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
    U_Y = S_dag @ H_gate
    
    rotation_map = {
        'I': I,
        'Z': I,      # No rotation for Z basis
        'X': H_gate, # Hadamard for X basis
        'Y': U_Y     # S†H for Y basis
    }
    
    OpList = []
    Sites = []
    
    for k, char in enumerate(basis_string):
        if char in rotation_map:
            OpList.append(rotation_map[char])
            Sites.append(k)
        else:
            raise ValueError(f"Unknown basis character: {char}")
    
    return Sites, reduce(np.kron, OpList)

def BuildBases(hilbert, basis_strings):
    """
    Build explicit LocalOperator objects for measurement bases.
    basis_strings: array of strings like ['XIYIZI', 'ZZIIIZ', ...]

    """
    base_ops = []
    
    for basis_string in basis_strings:
        basis_string = str(basis_string)  # Convert numpy.str_ if needed
        sites, operator = MeasurementRotationFromString(basis_string)
        base_operator = nk.operator.LocalOperator(hilbert, operator, sites)
        base_ops.append(base_operator)
    
    return np.array(base_ops, dtype=object)

def generate_beh2_measurement_dataset(
    ground_state,
    pauli_strings,
    coefficients,
    n_samples=128000,
    seed=42
):
    """
    Generate synthetic measurement dataset for BeH2.
    
    Args:
        ground_state: Exact ground state vector from diagonalization
        pauli_strings: List of Pauli strings from Hamiltonian decomposition
        coefficients: Corresponding coefficients
        n_samples: Number of measurements to generate
        seed: Random seed
        
    Returns:
        bitstrings: Array of measurement outcomes
        basis_strings: Array of measurement bases
        statistics: Dict with dataset statistics
    """
    np.random.seed(seed)
    n_qubits = int(np.log2(len(ground_state)))
    
    print(f"Generating BeH2 measurement dataset...")
    print(f"  Ground state: {len(ground_state)} dimensional")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Hamiltonian terms: {len(pauli_strings)}")
    print(f"  Target samples: {n_samples:,}")
    
    # Filter out identity terms (empty or all I's)
    non_identity_terms = []
    for pauli, coeff in zip(pauli_strings, coefficients):
        pauli_str = str(pauli)
        # Skip if empty or all I's
        if len(pauli_str) > 0 and not all(c == 'I' for c in pauli_str):
            non_identity_terms.append(pauli_str)
        elif len(pauli_str) == 0:
            print(f"  Skipping empty Pauli term with coefficient {coeff}")
    
    print(f"  Non-identity terms: {len(non_identity_terms)}")
    print(f"  Unique Pauli terms: {len(set(non_identity_terms))}")
    
    if len(non_identity_terms) == 0:
        raise ValueError("No non-identity Pauli terms found!")
    
    # Generate measurement samples
    bitstrings = []
    basis_strings = []
    basis_counter = Counter()
    
    print(f"\nGenerating {n_samples:,} measurement samples...")
    for i in range(n_samples):
        # Uniformly sample a Pauli term
        basis_string = np.random.choice(non_identity_terms)
        basis_counter[basis_string] += 1
        
        # Apply rotation and sample
        rotated_state = apply_measurement_rotation(ground_state, basis_string)
        bitstring = sample_bitstring(rotated_state)
        
        bitstrings.append(bitstring)
        basis_strings.append(basis_string)
        
        if (i + 1) % (n_samples // 10) == 0:
            print(f"  Progress: {i + 1:,}/{n_samples:,}")
    
    bitstrings = np.array(bitstrings)
    basis_strings = np.array(basis_strings)
    
    # Statistics
    unique_bases = len(set(basis_strings))
    unique_bitstrings = len(set(bitstrings))
    
    print(f"\n✓ Dataset generated!")
    print(f"  Total samples: {len(bitstrings):,}")
    print(f"  Unique bases used: {unique_bases}")
    print(f"  Unique bitstrings: {unique_bitstrings}")
    print(f"  Average samples per basis: {n_samples / unique_bases:.1f}")
    
    # Check coverage
    expected_unique = len(set(non_identity_terms))
    if unique_bases < expected_unique:
        missing = expected_unique - unique_bases
        print(f"  ⚠ Warning: Missing {missing} basis types (expected {expected_unique})")
    else:
        print(f"  ✓ All {expected_unique} unique basis types covered!")
    
    # Show top 5 most common bases
    print(f"\n  Top 5 most sampled bases:")
    for basis, count in basis_counter.most_common(5):
        print(f"    {basis}: {count} times ({100*count/n_samples:.2f}%)")
    
    statistics = {
        'n_samples': len(bitstrings),
        'unique_bases': unique_bases,
        'unique_bitstrings': unique_bitstrings,
        'expected_unique_bases': expected_unique,
        'basis_counts': dict(basis_counter)
    }
    
    return bitstrings, basis_strings, statistics


#%%
# Example usage
if __name__ == "__main__":
    print("="*60)
    print("BeH2 Measurement Dataset Generator")
    print("="*60)
    
    # Load BeH2 Hamiltonian data

    pauli_path = "../data/tomography/BeH2/paulis.file"
    coeff_path = "../data/tomography/BeH2/interactions.file"
    
    print("\nLoading BeH2 Hamiltonian...")
    pauli_strings = np.load(pauli_path, allow_pickle=True)
    coefficients = np.load(coeff_path, allow_pickle=True)
    
    print(f"  Loaded {len(pauli_strings)} Pauli terms")
    print(f"  Loaded {len(coefficients)} coefficients")
    
    # For now, create placeholder
    n_qubits = 6  # BeH2 has 6 qubits
    print(f"\nComputing ground state for N={n_qubits} qubits...")
    print("  (In practice, load from exact diagonalization)")
    
    # Placeholder ground state - replace with actual
    psi0 = np.loadtxt("../data/tomography/BeH2/psi.txt")
    
    # Generate dataset
    print("\n" + "="*60)
    bitstrings, basis_strings, stats = generate_beh2_measurement_dataset(
        psi0,
        pauli_strings,
        coefficients,
        n_samples=128000,
        seed=42
    )
    
    # Convert bitstrings to array format for QSR
    bitstring_array = np.array([list(bs) for bs in bitstrings], dtype=float)
    
    # Build explicit LocalOperator objects for bases
    print("\n" + "="*60)
    print("Building explicit basis operators...")
    hilbert = nk.hilbert.Spin(s=0.5, N=n_qubits)
    basis_operators = BuildBases(hilbert, basis_strings)
    print(f"  ✓ Created {len(basis_operators)} basis operators")
    
    # Verify one operator
    print("\n  Testing basis operator:")
    test_op = basis_operators[0]
    print(f"    {test_op}")
    
    # Save dataset
    output_file = f'beh2_measurements_M{len(bitstrings)}new.npz'
    print(f"\nSaving dataset to {output_file}...")
    np.savez(output_file,
             bitstrings=bitstring_array,
             basis_strings=basis_strings,
             basis_operators=basis_operators,
             ground_state=psi0,
             statistics=stats,
             pauli_strings=pauli_strings,
             coefficients=coefficients,
             n_qubits=n_qubits)
    
    print(f"✓ Dataset saved!")
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
# %%
