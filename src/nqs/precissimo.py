import numpy as np
import matplotlib.pyplot as plt

# A. The neural network
class RestrictedBoltzmannMachine:
    """
    RBM with binary spins σ_i ∈ {-1, +1} for visible layer
    and h_j ∈ {-1, +1} for hidden layer.
    
    COMPLEX PARAMETERS: a_i, b_j, W_ij ∈ ℂ
    
    The RBM wavefunction is:
    ψ_λ(σ) = exp(Σ_i a_i σ_i) × ∏_j 2cosh(θ_j)
    
    where θ_j = b_j + Σ_i W_ij σ_i
    
    The probability distribution for sampling is:
    P(σ) = |ψ_λ(σ)|² / Z
    """
    
    def __init__(self, n_visible, n_hidden, seed=42):
        """
        Initialize RBM with random complex parameters.
        
        Parameters:
        -----------
        n_visible : int
            Number of visible units (spins)
        n_hidden : int
            Number of hidden units
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        
        # Initialize COMPLEX parameters
        # Each parameter has real and imaginary parts
        scale = 0.01
        
        # Visible biases: a_i ∈ ℂ
        self.a = scale * (self.rng.standard_normal(n_visible) + 
                         1j * self.rng.standard_normal(n_visible))
        
        # Hidden biases: b_j ∈ ℂ
        self.b = scale * (self.rng.standard_normal(n_hidden) + 
                         1j * self.rng.standard_normal(n_hidden))
        
        # Weights: W_ij ∈ ℂ
        self.W = scale * (self.rng.standard_normal((n_visible, n_hidden)) + 
                         1j * self.rng.standard_normal((n_visible, n_hidden)))
        
        print(f"Created Complex RBM with {n_visible} visible and {n_hidden} hidden units")
        print(f"Total parameters: {self.count_parameters()}")
        print(f"  (Each parameter has real and imaginary parts)")
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        # Each complex parameter counts as 2 real parameters
        return 2 * (self.n_visible + self.n_hidden + self.n_visible * self.n_hidden)
    
    def energy(self, sigma, h): # we don't really need this part: just for pedagogical purposes )
        """
        Compute the energy E(σ, h) for a configuration.
        
        E(σ, h) = -Σ_i a_i σ_i - Σ_j b_j h_j - Σ_ij W_ij σ_i h_j
        
        Note: Even with complex parameters, the energy in the exponential
        is used to define the probability, so we work with the wavefunction directly.
        """
        visible_term = -np.dot(self.a, sigma)
        hidden_term = -np.dot(self.b, h)
        interaction_term = -np.dot(sigma, np.dot(self.W, h))
        
        return visible_term + hidden_term + interaction_term
    
    def log_wavefunction(self, sigma):
        """
        Compute log ψ_λ(σ) for numerical stability.
        
        log ψ_λ(σ) = Σ_i a_i σ_i + Σ_j log(2cosh(θ_j))
        
        where θ_j = b_j + Σ_i W_ij σ_i
        
        Returns complex number.
        """
        # Visible layer contribution
        visible_term = np.dot(self.a, sigma)
        
        # Hidden layer contribution (after marginalizing)
        theta = self.b + np.dot(sigma, self.W)
        hidden_term = np.sum(np.log(2.0 * np.cosh(theta)))
        
        return visible_term + hidden_term
    
    def wavefunction(self, sigma):
        """
        Compute the complex wavefunction ψ_λ(σ).
        
        ψ_λ(σ) = exp(Σ_i a_i σ_i) × ∏_j 2cosh(θ_j)
        
        Returns: complex number
        """
        return np.exp(self.log_wavefunction(sigma))
    
    def probability_unnormalized(self, sigma):
        """
        Compute unnormalized probability P(σ) ∝ |ψ_λ(σ)|².
        
        This is what we sample from!
        
        Returns: real number (probability)
        """
        psi = self.wavefunction(sigma)
        return np.abs(psi)**2  # |ψ|² = ψ* × ψ
    
    def log_probability_unnormalized(self, sigma):
        """
        Compute log|ψ_λ(σ)|² = 2 Re[log ψ_λ(σ)] for numerical stability.
        
        This is useful for Metropolis-Hastings to avoid overflow.
        """
        log_psi = self.log_wavefunction(sigma)
        return 2 * np.real(log_psi)
    
    
# B. The MC Sampler
class RBMSampler(RestrictedBoltzmannMachine):
    """Extends Complex RBM with sampling methods."""
    
    def metropolis_hastings_step(self, sigma):
        """
        One step of Metropolis-Hastings: propose flipping one random spin.
        NOW WORKS FOR {0, 1} BASIS )
        
        For complex RBM:
        Acceptance ratio: R = |ψ(σ')|² / |ψ(σ)|²
        
        Using log probabilities for numerical stability:
        log R = 2[Re(log ψ(σ')) - Re(log ψ(σ))]
        """
        # Current log probability
        log_prob_old = self.log_probability_unnormalized(sigma)
        
        # Propose flipping a random spin
        i = self.rng.integers(self.n_visible)
        sigma_prime = sigma.copy()
        sigma_prime[i] = 1 - sigma_prime[i]
        
        # New log probability
        log_prob_new = self.log_probability_unnormalized(sigma_prime)
        
        # Log acceptance ratio --- at this step Z would cancel so we don't even need it!
        log_R = log_prob_new - log_prob_old
        
        # Acceptance probability A = min(1, R)
        A = min(1.0, np.exp(log_R))
        
        # Accept with probability A
        if self.rng.random() < A:
            return sigma_prime, True  # accepted
        else:
            return sigma, False  # rejected
    
    def metropolis_hastings_step_fast(self, sigma, log_prob_old, theta_old):
        """
        Optimized version that reuses computed quantities.
        Works for 0,1 basis now!
        
        Parameters:
        -----------
        sigma : array
            Current configuration
        log_prob_old : float
            Pre-computed log|ψ(σ)|²
        theta_old : array (complex)
            Pre-computed θ_j = b_j + Σ_i W_ij σ_i
            
        Returns:
        --------
        sigma_new, accepted, log_prob_new, theta_new
        """
        # Propose flipping a random spin
        i = self.rng.integers(self.n_visible)
        sigma_prime = sigma.copy()
        sigma_prime[i] = 1 - sigma_prime[i]
        
        # Update theta efficiently: θ'_j = θ_j + W_ij * (new_sigma_i - old_sigma_i)
        delta_sigma = sigma_prime[i] - sigma[i]  # Either +1 or -1
        theta_new = theta_old + self.W[i, :] * delta_sigma
        
        # Compute new log probability
        visible_term = np.dot(self.a, sigma_prime)
        hidden_term = np.sum(np.log(2.0 * np.cosh(theta_new)))
        log_psi_new = visible_term + hidden_term
        log_prob_new = 2 * np.real(log_psi_new)
        
        # Log acceptance ratio
        log_R = log_prob_new - log_prob_old
        
        # Acceptance probability A = min(1, R)
        A = min(1.0, np.exp(log_R))
        
        # Accept with probability A
        if self.rng.random() < A:
            return sigma_prime, True, log_prob_new, theta_new  # accepted
        else:
            return sigma, False, log_prob_old, theta_old  # rejected
    
    
    def sample_from_distribution(self, n_samples, burn_in=1000, thin=10, 
                                 verbose=True, use_fast=True):
        """
        Generate samples from P(σ) = |ψ_λ(σ)|² using Metropolis-Hastings.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to collect
        burn_in : int
            Number of initial steps to discard
        thin : int
            Keep only every thin-th sample
        use_fast : bool
            Use optimized version with theta caching
        """
        if verbose:
            print(f"\nStarting MCMC sampling from |ψ_λ|²...")
            print(f"  Burn-in: {burn_in} steps")
            print(f"  Collecting: {n_samples} samples")
            print(f"  Thinning: every {thin} steps")
            print(f"  Total MCMC steps: {burn_in + n_samples * thin}")
        
        # Start from random configuration
        sigma = self.rng.integers(0, 2, size=self.n_visible)
        
        if use_fast:
            # Pre-compute for fast updates
            log_prob = self.log_probability_unnormalized(sigma)
            theta = self.b + np.dot(sigma, self.W)
        
        samples = []
        n_accepted = 0
        total_steps = burn_in + n_samples * thin
        
        for step in range(total_steps):
            if use_fast:
                sigma, accepted, log_prob, theta = self.metropolis_hastings_step_fast(
                    sigma, log_prob, theta
                )
            else:
                sigma, accepted = self.metropolis_hastings_step(sigma)
            
            if accepted:
                n_accepted += 1
            
            # After burn-in, collect samples with thinning
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples.append(sigma.copy())
            
            # Progress indicator
            if verbose and (step + 1) % 2000 == 0:
                acceptance_rate = n_accepted / (step + 1)
                print(f"  Step {step + 1}/{total_steps}, acceptance rate: {acceptance_rate:.3f}")
        
        acceptance_rate = n_accepted / total_steps
        if verbose:
            print(f"\n✓ Sampling complete!")
            print(f"  Final acceptance rate: {acceptance_rate:.3f}")
            print(f"  (Optimal range: 0.2 - 0.5 for Metropolis-Hastings)")
        
        return np.array(samples)
    






# D. OBSERVABLE ESTIMATION - from measurement data (obtained elsewhere)

# ============================================
# 1. Pauli Operator Application
# ============================================

def apply_pauli_string(sigma_01, pauli_string):
    """
    Apply Pauli string in {0,1} basis.
    
    Args:
        sigma_01: array with values in {0, 1}
        pauli_string: e.g., 'XZIY'
    
    Returns:
        sigma_new: transformed config in {0,1}
        phase: complex phase
    """
    sigma_new = sigma_01.copy()
    phase = 1.0 + 0.0j
    
    for i, pauli in enumerate(pauli_string):
        if pauli == 'I':
            continue
        elif pauli == 'Z':
            # Z: eigenvalue = (-1)^v
            phase *= (-1)**sigma_01[i]
        elif pauli == 'X':
            # X: bit flip 0↔1
            sigma_new[i] = 1 - sigma_new[i]
        elif pauli == 'Y':
            # Y = iXZ
            phase *= 1j * (-1)**sigma_01[i]
            sigma_new[i] = 1 - sigma_new[i]
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")
    
    return sigma_new, phase


# ============================================
# 2. Local Estimator for One Sample
# ============================================

def compute_local_estimator(rbm, sigma, pauli_terms):
    """
    Compute the local estimator O_loc(σ) for a given configuration.
    
    Args:
        rbm: Your RBMSampler object with trained parameters
        sigma: spin configuration, shape (n_visible,) with values in {-1, +1}
        pauli_terms: list of tuples (coefficient, pauli_string)
                     e.g., [(0.5, 'XXII'), (-0.3, 'ZZZI'), (1.2, 'IYXI')]
    
    Returns:
        O_loc: complex number (local estimator value)
    """
    # Compute ψ(σ)
    psi_sigma = rbm.wavefunction(sigma)  # Using your RBM's psi method
    
    O_loc = 0.0 + 0.0j
    
    for coeff, pauli_string in pauli_terms:
        # Apply Pauli string to get σ_k and phase
        sigma_k, pauli_phase = apply_pauli_string(sigma, pauli_string)
        
        # Compute ψ(σ_k)
        psi_sigma_k = rbm.wavefunction(sigma_k)
        
        # Add contribution: c_k * (ψ(σ_k) / ψ(σ)) * ⟨σ|P_k|σ_k⟩
        # Note: ⟨σ|P_k|σ_k⟩ = pauli_phase
        O_loc += coeff * pauli_phase * (psi_sigma_k / psi_sigma)
    
    return O_loc


# ============================================
# 3. Monte Carlo Estimation of Observable
# ============================================

def estimate_observable(rbm, pauli_terms, n_samples=10000, burn_in=2000, thin=10):
    """
    Estimate ⟨O⟩ using Monte Carlo sampling from the trained RBM.
    
    Args:
        rbm: Trained RBMSampler
        pauli_terms: Observable as list of (coefficient, pauli_string) tuples
        n_samples: Number of MC samples
        burn_in: Burn-in steps for sampling
        thin: Thinning interval
    
    Returns:
        expectation: ⟨O⟩ estimate
        std_error: Standard error of the estimate
        all_local_values: Array of all O_loc values (for diagnostics)
    """
    print(f"Sampling {n_samples} configurations from trained RBM...")
    
    # Sample configurations from |ψ_λ|²
    samples = rbm.sample_from_distribution(
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin
    )
    
    print(f"Computing local estimators...")
    
    # Compute O_loc for each sample
    local_values = []
    for i, sigma in enumerate(samples):
        if i % 1000 == 0:
            print(f"  Processing sample {i}/{n_samples}...")
        
        O_loc = compute_local_estimator(rbm, sigma, pauli_terms)
        local_values.append(O_loc)
    
    local_values = np.array(local_values)
    
    # Compute expectation and standard error
    expectation = np.mean(local_values)
    std_error = np.std(local_values) / np.sqrt(len(local_values))
    
    return expectation, std_error, local_values

