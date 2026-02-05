"""
Shadow Hamiltonian Simulation POC - Direct Fourier Evolution.

For free particle H = k²/2 with periodic BC, the wavefunction evolves exactly as:
  psi_k(t) = psi_k(0) * exp(-i k² t / 2)

This is the SAME Hamiltonian as circuit_2D.py uses (via QFT).

The "shadow" here is that we evolve in Fourier space directly (classical FFT)
instead of using quantum circuits. This gives EXACT results for the free particle.

For observables like density Fourier modes, we:
1. Evolve psi(t) using FFT-based propagation
2. Compute rho(t) = |psi(t)|²
3. Extract Fourier features from rho(t)

This demonstrates that shadow simulation can be exact when the observable
dynamics can be computed efficiently classically.

Run with: python -m poc.run_poc (from test/ directory)
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Support both module and direct execution
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from poc.baseline import evolve
    from poc.init_states import create_gaussian_wavepacket
else:
    from .baseline import evolve
    from .init_states import create_gaussian_wavepacket


def evolve_wavefunction_fourier(psi0, N, t):
    """
    Evolve wavefunction in Fourier space for free particle H = k²/2.

    This implements the SAME evolution as circuit_2D.py but classically.

    Args:
        psi0: Initial wavefunction (N*N,) or (N, N)
        N: Grid size
        t: Time

    Returns:
        psi_t: Evolved wavefunction (N, N)
    """
    psi = psi0.reshape(N, N)

    # Transform to Fourier space
    psi_k = np.fft.fft2(psi)

    # Momentum grid (matching circuit_2D.py convention)
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)

    # Kinetic energy: k² = kx² + ky²
    K2 = KX**2 + KY**2

    # Time evolution phase: exp(-i H t) = exp(-i k²/2 * t)
    phase = np.exp(-0.5j * K2 * t)

    # Evolve in Fourier space
    psi_k_t = psi_k * phase

    # Transform back
    psi_t = np.fft.ifft2(psi_k_t)

    return psi_t


def extract_density(psi, N):
    """Extract density rho = |psi|² from wavefunction."""
    return np.abs(psi.reshape(N, N))**2


def extract_fourier_block(rho, K):
    """
    Extract low-frequency Fourier block from density.

    Returns:
        block: (2K+1, 2K+1) complex Fourier coefficients
        energy: sum(|block|²)
    """
    N = rho.shape[0]
    rho_hat = np.fft.fft2(rho)
    rho_hat_c = np.fft.fftshift(rho_hat)

    center = N // 2
    block = rho_hat_c[center-K:center+K+1, center-K:center+K+1]
    energy = np.sum(np.abs(block)**2)

    return block, energy


def extract_features(block):
    """Convert complex block to real feature vector."""
    vec = block.flatten()
    return np.concatenate([np.real(vec), np.imag(vec)])


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================
    nx = 4
    ny = 4
    N = 2**nx  # 16
    K = 2  # Fourier block half-width

    # Initial state parameters
    sigma = 0.8
    x0 = 0.0
    y0 = 0.0
    k0x = 2.0
    k0y = 1.0

    # Time points
    t_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dt = 0.1

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Shadow Hamiltonian Simulation POC - Direct Fourier Evolution")
    print("=" * 70)
    print(f"Grid: {N}x{N}, qubits: nx={nx}, ny={ny}")
    print(f"Fourier block: K={K}, size={(2*K+1)}x{(2*K+1)}")
    print(f"Time points: {len(t_list)}, dt={dt}")
    print()
    print("Method: Direct Fourier-space evolution (same H as baseline)")
    print("  H = k²/2 (free particle, periodic BC)")
    print("  psi_k(t) = psi_k(0) * exp(-i k² t / 2)")
    print()

    # ==========================================================================
    # Create initial state
    # ==========================================================================
    print("Creating initial state (Gaussian wavepacket)...")
    initial_state_full = create_gaussian_wavepacket(
        nx, ny, sigma=sigma, x0=x0, y0=y0, k0x=k0x, k0y=k0y
    )
    psi0 = initial_state_full[:N*N]
    print(f"  |psi0| norm: {np.linalg.norm(psi0):.6f}")
    print()

    # ==========================================================================
    # Baseline: Full quantum circuit evolution
    # ==========================================================================
    print("Running baseline quantum circuit evolution...")
    baseline_start = time.time()

    statevectors_baseline = []
    for t in t_list:
        if t == 0.0:
            sv = initial_state_full.copy()
        else:
            sv = evolve(nx, ny, dt=t, initial_state=initial_state_full)
        statevectors_baseline.append(sv)

    baseline_time = time.time() - baseline_start
    print(f"  Baseline completed in {baseline_time:.3f} seconds")

    # Extract densities and features from baseline
    densities_baseline = []
    blocks_baseline = []
    energies_baseline = []
    features_baseline = []

    for sv in statevectors_baseline:
        psi = sv[:N*N]
        rho = extract_density(psi, N)
        block, energy = extract_fourier_block(rho, K)
        features = extract_features(block)

        densities_baseline.append(rho)
        blocks_baseline.append(block)
        energies_baseline.append(energy)
        features_baseline.append(features)

    print()

    # ==========================================================================
    # Shadow: Direct Fourier evolution (classical)
    # ==========================================================================
    print("Running shadow simulation (direct Fourier evolution)...")
    shadow_start = time.time()

    densities_shadow = []
    blocks_shadow = []
    energies_shadow = []
    features_shadow = []

    for t in t_list:
        if t == 0.0:
            psi_t = psi0.reshape(N, N)
        else:
            psi_t = evolve_wavefunction_fourier(psi0, N, t)

        rho = extract_density(psi_t, N)
        block, energy = extract_fourier_block(rho, K)
        features = extract_features(block)

        densities_shadow.append(rho)
        blocks_shadow.append(block)
        energies_shadow.append(energy)
        features_shadow.append(features)

    shadow_time = time.time() - shadow_start
    print(f"  Shadow completed in {shadow_time:.6f} seconds")
    print()

    # ==========================================================================
    # Compute errors
    # ==========================================================================
    print("Computing errors (shadow vs baseline)...")

    # Feature errors
    feature_errors = []
    print("  Feature relative errors:")
    for i, t in enumerate(t_list):
        f_base = features_baseline[i]
        f_shadow = features_shadow[i]
        rel_err = np.linalg.norm(f_shadow - f_base) / (np.linalg.norm(f_base) + 1e-10)
        feature_errors.append(rel_err)
        print(f"    t={t:.1f}: {rel_err:.6e}")

    # Density errors
    density_errors = []
    print()
    print("  Density L2 relative errors:")
    for i, t in enumerate(t_list):
        rho_base = densities_baseline[i]
        rho_shadow = densities_shadow[i]
        rel_err = np.linalg.norm(rho_shadow - rho_base) / (np.linalg.norm(rho_base) + 1e-10)
        density_errors.append(rel_err)
        print(f"    t={t:.1f}: {rel_err:.6e}")

    # Energy errors
    energy_errors = []
    print()
    print("  Low-frequency energy relative errors:")
    for i, t in enumerate(t_list):
        E_base = energies_baseline[i]
        E_shadow = energies_shadow[i]
        rel_err = np.abs(E_shadow - E_base) / (E_base + 1e-10)
        energy_errors.append(rel_err)
        print(f"    t={t:.1f}: {rel_err:.6e}")

    # ==========================================================================
    # Save outputs
    # ==========================================================================
    print()
    print("Saving outputs...")

    np.save(os.path.join(output_dir, 'features_baseline.npy'), np.array(features_baseline))
    np.save(os.path.join(output_dir, 'features_shadow.npy'), np.array(features_shadow))
    print(f"  Saved features")

    # Plot: Errors over time
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(t_list, feature_errors, 'o-', color='blue', linewidth=2, markersize=6)
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Relative Error')
    axes[0].set_title('Feature Error')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_list, density_errors, 's-', color='green', linewidth=2, markersize=6)
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Density L2 Error')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_list, energy_errors, '^-', color='red', linewidth=2, markersize=6)
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('Relative Error')
    axes[2].set_title('Low-Freq Energy Error')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'errors_over_time.png'), dpi=150)
    plt.close()
    print(f"  Saved errors_over_time.png")

    # Plot: Density comparison
    from numpy import pi
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    time_indices = [0, 2, 4, 6, 9]

    for col, t_idx in enumerate(time_indices):
        t_val = t_list[t_idx]
        vmax = max(densities_baseline[t_idx].max(), densities_shadow[t_idx].max())

        im0 = axes[0, col].pcolormesh(X, Y, densities_baseline[t_idx], cmap='viridis',
                                       vmin=0, vmax=vmax, shading='auto')
        axes[0, col].set_title(f't={t_val}')
        axes[0, col].set_aspect('equal')
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel('Baseline (QC)', fontsize=11)

        im1 = axes[1, col].pcolormesh(X, Y, densities_shadow[t_idx], cmap='viridis',
                                       vmin=0, vmax=vmax, shading='auto')
        axes[1, col].set_aspect('equal')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel('Shadow (FFT)', fontsize=11)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label=r'$\rho$')

    plt.suptitle('Density: Baseline (Quantum Circuit) vs Shadow (Direct FFT)', fontsize=12, y=0.98)
    plt.savefig(os.path.join(output_dir, 'density_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved density_comparison.png")

    # Plot: Energy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_list, energies_baseline, 'o-', color='blue', linewidth=2, markersize=8, label='Baseline (QC)')
    ax.plot(t_list, energies_shadow, 's--', color='red', linewidth=2, markersize=6, label='Shadow (FFT)')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Low-Frequency Energy')
    ax.set_title('Low-Frequency Fourier Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_over_time.png'), dpi=150)
    plt.close()
    print(f"  Saved energy_over_time.png")

    # Runtime summary
    speedup = baseline_time / shadow_time if shadow_time > 0 else float('inf')
    runtime_summary = f"""Shadow Hamiltonian Simulation POC - Direct Fourier Evolution
============================================================

Method:
  Shadow simulation uses direct Fourier-space evolution:
    psi_k(t) = psi_k(0) * exp(-i k² t / 2)
  This is the SAME Hamiltonian H = k²/2 as the baseline quantum circuit.
  The shadow is "exact" for this free-particle case.

Configuration:
  Grid size: {N}x{N}
  Qubits: nx={nx}, ny={ny}
  Fourier block: K={K}, size={(2*K+1)}x{(2*K+1)}
  Feature dimension: {len(features_baseline[0])}
  Time points: {len(t_list)}, dt={dt}

Runtime:
  Baseline (quantum circuit): {baseline_time:.3f} seconds
  Shadow (direct FFT): {shadow_time:.6f} seconds
  Speedup: {speedup:.0f}x

Errors (shadow vs baseline):
  Mean feature error: {np.mean(feature_errors):.6e}
  Mean density error: {np.mean(density_errors):.6e}
  Mean energy error: {np.mean(energy_errors):.6e}
  Max feature error: {np.max(feature_errors):.6e}
  Max density error: {np.max(density_errors):.6e}
  Max energy error: {np.max(energy_errors):.6e}

Note:
  Small errors are due to numerical precision differences between
  quantum circuit simulation and direct FFT computation.
"""

    with open(os.path.join(output_dir, 'runtime_summary.txt'), 'w') as f:
        f.write(runtime_summary)
    print(f"  Saved runtime_summary.txt")

    print()
    print("=" * 70)
    print("POC completed successfully!")
    print("=" * 70)
    print(f"Baseline time: {baseline_time:.3f}s, Shadow time: {shadow_time:.6f}s")
    print(f"Speedup: {speedup:.0f}x")
    print(f"Mean feature error: {np.mean(feature_errors):.6e}")
    print(f"Mean density error: {np.mean(density_errors):.6e}")


if __name__ == '__main__':
    main()
