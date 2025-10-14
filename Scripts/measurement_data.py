import numpy as np
from typing import List, Tuple
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info import Operator

class MeasurementDatasetGenerator:
    """
    Generate synthetic measurement datasets for quantum state tomography.
    
    This simulates the measurement process on quantum hardware by:
    1. Starting with an exact ground state wavefunction
    2. Decomposing the Hamiltonian into Pauli strings
    3. Randomly selecting measurement bases from the Pauli decomposition
    4. Rotating the wavefunction and sampling measurement outcomes
    """
    
    def __init__(self, hamiltonian: SparsePauliOp, ground_state: Statevector):
        """
        Initialize the dataset generator.
        
        Args:
            hamiltonian: Hamiltonian as SparsePauliOp (sum of Pauli strings)
            ground_state: Exact ground state wavefunction
        """
        self.hamiltonian = hamiltonian
        self.ground_state = ground_state
        self.n_qubits = ground_state.num_qubits
        
        # Extract Pauli strings from Hamiltonian
        self.pauli_strings = self._extract_pauli_strings()
        
        print(f"Initialized generator for {self.n_qubits} qubits")
        print(f"Hamiltonian contains {len(self.pauli_strings)} Pauli terms")
    
    def _extract_pauli_strings(self) -> List[str]:
        """
        STEP 1: Extract Pauli strings from Hamiltonian decomposition.
        
        Returns:
            List of Pauli strings (e.g., ['IXYZ', 'ZZII', ...])
        """
        pauli_list = []
        
        # SparsePauliOp has .paulis attribute containing the Pauli terms
        for pauli in self.hamiltonian.paulis:
            pauli_str = pauli.to_label()
            pauli_list.append(pauli_str)
        
        print(f"\nExample Pauli strings:")
        for p in pauli_list[:5]:
            print(f"  {p}")
        
        return pauli_list
    
    def _pauli_string_to_basis(self, pauli_string: str) -> List[str]:
        """
        STEP 2: Convert Pauli string to measurement bases.
        
        Args:
            pauli_string: e.g., 'XZIZY' 
        
        Returns:
            List of basis for each qubit: ['X', 'Z', 'I', 'Z', 'Y']
        """
        # Qiskit uses reverse ordering (rightmost = qubit 0)
        # Reverse to get qubit 0 first
        return list(reversed(pauli_string))
    
    def _get_rotation_for_basis(self, basis: str) -> np.ndarray:
        """
        STEP 3: Get unitary rotation matrix for measuring in a given basis.
        
        The rotation transforms from computational (Z) basis to the target basis.
        
        Args:
            basis: 'X', 'Y', 'Z', or 'I'
        
        Returns:
            2x2 unitary matrix
        """
        if basis == 'Z' or basis == 'I':
            # No rotation needed for Z basis
            return np.eye(2, dtype=complex)
        
        elif basis == 'X':
            # Hadamard gate: Z -> X
            return np.array([[1, 1], 
                           [1, -1]], dtype=complex) / np.sqrt(2)
        
        elif basis == 'Y':
            # S†H gate: Z -> Y
            H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            S_dag = np.array([[1, 0], [0, -1j]], dtype=complex)
            return S_dag @ H
        
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def _rotate_state(self, bases: List[str]) -> Statevector:
        """
        STEP 4: Rotate the ground state according to measurement bases.
        
        Applies U(τ) = ⊗ⱼ U(τⱼ) where U(τⱼ) rotates qubit j to basis τⱼ.
        
        Args:
            bases: Measurement basis for each qubit
        
        Returns:
            Rotated state vector
        """
        # Start with the ground state amplitudes
        psi = self.ground_state.data.copy()
        
        # Apply rotation to each qubit
        for qubit_idx, basis in enumerate(bases):
            U = self._get_rotation_for_basis(basis)
            psi = self._apply_single_qubit_gate(psi, U, qubit_idx)
        
        return Statevector(psi)
    
    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, 
                                  qubit: int) -> np.ndarray:
        """
        Apply a single-qubit gate to a specific qubit in the state vector.
        
        Args:
            state: State vector (length 2^n)
            gate: 2x2 unitary matrix
            qubit: Which qubit to apply gate to (0 = first qubit)
        
        Returns:
            New state vector after applying gate
        """
        n = self.n_qubits
        d = 2**n
        
        # Reshape state vector into tensor with one index per qubit
        # Shape: (2, 2, 2, ..., 2) with n dimensions
        psi_tensor = state.reshape([2] * n)
        
        # Apply gate to the specified qubit
        # This contracts gate[a,b] with psi_tensor[..., b, ...]
        psi_tensor = np.tensordot(gate, psi_tensor, axes=([1], [qubit]))
        
        # Move the contracted index back to the correct position
        psi_tensor = np.moveaxis(psi_tensor, 0, qubit)
        
        # Flatten back to state vector
        return psi_tensor.reshape(d)
    
    def _sample_measurement(self, rotated_state: Statevector) -> str:
        """
        STEP 5: Sample a measurement outcome from the rotated state.
        
        Samples from P(σ) = |⟨σ|ψ_rotated⟩|²
        
        Args:
            rotated_state: State after rotation to measurement basis
        
        Returns:
            Measurement outcome as bit string (e.g., '010110')
        """
        # Get probabilities for all computational basis states
        probabilities = rotated_state.probabilities()
        
        # Sample an outcome
        outcome_index = np.random.choice(len(probabilities), p=probabilities)
        
        # Convert to binary string
        bitstring = format(outcome_index, f'0{self.n_qubits}b')
        
        return bitstring
    
    def generate_single_measurement(self) -> Tuple[List[str], str]:
        """
        Generate a single measurement (basis, outcome) pair.
        
        Returns:
            (bases, outcome) where:
                bases: List of measurement basis for each qubit
                outcome: Bit string of measurement result
        """
        # STEP 1: Randomly select a Pauli string from Hamiltonian
        pauli_string = np.random.choice(self.pauli_strings)
        
        # STEP 2: Convert to measurement bases
        bases = self._pauli_string_to_basis(pauli_string)
        
        # STEP 3 & 4: Rotate state to measurement basis
        rotated_state = self._rotate_state(bases)
        
        # STEP 5: Sample measurement outcome
        outcome = self._sample_measurement(rotated_state)
        
        return bases, outcome
    
    def generate_dataset(self, n_measurements: int, 
                        verbose: bool = True) -> List[Tuple[List[str], str]]:
        """
        Generate a complete measurement dataset.
        
        Args:
            n_measurements: Number of measurements to generate
            verbose: Whether to print progress
        
        Returns:
            List of (bases, outcome) tuples representing measurements
        """
        dataset = []
        
        if verbose:
            print(f"\nGenerating {n_measurements} measurements...")
            print("=" * 60)
        
        for i in range(n_measurements):
            bases, outcome = self.generate_single_measurement()
            dataset.append((bases, outcome))
            
            if verbose and (i + 1) % max(1, n_measurements // 10) == 0:
                print(f"Progress: {i + 1}/{n_measurements} measurements")
        
        if verbose:
            print("=" * 60)
            print(f"Dataset generation complete!\n")
            self._print_dataset_summary(dataset)
        
        return dataset
    
    def _print_dataset_summary(self, dataset: List[Tuple[List[str], str]]):
        """Print summary statistics about the generated dataset."""
        print("\nDataset Summary:")
        print(f"  Total measurements: {len(dataset)}")
        print(f"  Qubits per measurement: {self.n_qubits}")
        
        # Count basis usage
        from collections import Counter
        all_bases = []
        for bases, _ in dataset:
            all_bases.extend(bases)
        basis_counts = Counter(all_bases)
        
        print(f"\n  Basis usage:")
        for basis in ['X', 'Y', 'Z', 'I']:
            count = basis_counts.get(basis, 0)
            percentage = 100 * count / len(all_bases) if all_bases else 0
            print(f"    {basis}: {count:6d} ({percentage:5.1f}%)")
        
        print(f"\n  First 5 measurements:")
        for i, (bases, outcome) in enumerate(dataset[:5]):
            bases_str = ''.join(bases)
            print(f"    {i+1}. Bases: {bases_str}, Outcome: {outcome}")
    
    def save_dataset(self, dataset: List[Tuple[List[str], str]], 
                     filename: str):
        """
        Save dataset to file.
        
        Format: Each line contains "BASES OUTCOME"
        Example: "XZIY 0101"
        
        Args:
            dataset: Generated measurement dataset
            filename: Path to save file
        """
        with open(filename, 'w') as f:
            for bases, outcome in dataset:
                bases_str = ''.join(bases)
                f.write(f"{bases_str} {outcome}\n")
        
        print(f"Dataset saved to {filename}")
    
    @staticmethod
    def load_dataset(filename: str) -> List[Tuple[List[str], str]]:
        """
        Load dataset from file.
        
        Args:
            filename: Path to dataset file
        
        Returns:
            List of (bases, outcome) tuples
        """
        dataset = []
        with open(filename, 'r') as f:
            for line in f:
                bases_str, outcome = line.strip().split()
                bases = list(bases_str)
                dataset.append((bases, outcome))
        
        print(f"Loaded {len(dataset)} measurements from {filename}")
        return dataset


# ============================================================================
# EXAMPLE USAGE - UPDATED FOR NEW QISKIT API
# ============================================================================

def example_simple_hamiltonian():
    """
    Simple example with a manually defined Hamiltonian.
    Uses new Qiskit 1.0+ API.
    """
    from qiskit.quantum_info import SparsePauliOp
    
    print("Creating simple 2-qubit Hamiltonian...")
    
    # H = 0.5*ZZ + 0.3*XX + 0.2*XI
    # New syntax: SparsePauliOp(pauli_list, coeffs)
    hamiltonian = SparsePauliOp(
        ['ZZ', 'XX', 'XI'],
        coeffs=[0.5, 0.3, 0.2]
    )
    
    print("Hamiltonian:", hamiltonian)
    
    # Get ground state using exact diagonalization
    from scipy.sparse.linalg import eigsh
    
    # Convert to matrix
    H_matrix = hamiltonian.to_matrix()
    
    # Find ground state
    eigenvalues, eigenvectors = eigsh(H_matrix, k=1, which='SA')
    ground_energy = eigenvalues[0]
    ground_state = Statevector(eigenvectors[:, 0])
    
    print(f"Ground state energy: {ground_energy:.6f}")
    print(f"Number of qubits: {ground_state.num_qubits}")
    
    # Generate measurements
    generator = MeasurementDatasetGenerator(hamiltonian, ground_state)
    dataset = generator.generate_dataset(n_measurements=1000)
    
    return generator, dataset


def example_4qubit_hamiltonian():
    """
    Slightly larger 4-qubit example.
    """
    from qiskit.quantum_info import SparsePauliOp
    
    print("Creating 4-qubit Hamiltonian...")
    
    # Larger Hamiltonian with more terms
    hamiltonian = SparsePauliOp(
        ['ZZZZ', 'XXII', 'IIXX', 'XYZI', 'ZIXY', 'YYYY'],
        coeffs=[1.0, 0.5, 0.5, 0.3, 0.3, 0.2]
    )
    
    print(f"Hamiltonian has {len(hamiltonian.paulis)} terms")
    
    # Get ground state
    from scipy.sparse.linalg import eigsh
    H_matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = eigsh(H_matrix, k=1, which='SA')
    
    ground_energy = eigenvalues[0]
    ground_state = Statevector(eigenvectors[:, 0])
    
    print(f"Ground state energy: {ground_energy:.6f}")
    
    # Generate measurements
    generator = MeasurementDatasetGenerator(hamiltonian, ground_state)
    dataset = generator.generate_dataset(n_measurements=5000)
    
    # Save dataset
    generator.save_dataset(dataset, "4qubit_measurements.txt")
    
    return generator, dataset


if __name__ == "__main__":
    # Run simple 2-qubit example
    print("Running simple 2-qubit example...")
    print("=" * 60)
    generator, dataset = example_simple_hamiltonian()
    
    print("\n\n" + "=" * 60)
    print("Running 4-qubit example...")
    print("=" * 60)
    generator4, dataset4 = example_4qubit_hamiltonian()
    
    # Uncomment to run BeH2 example (requires qiskit-nature)
    # print("\n\n" + "=" * 60)
    # print("Running BeH2 molecule example...")
    # print("=" * 60)
    # generator, dataset = example_beh2_molecule()