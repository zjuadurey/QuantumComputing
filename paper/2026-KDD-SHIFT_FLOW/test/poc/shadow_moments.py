"""
Shadow Hamiltonian Simulation with Moment Observables.

For free particle H = P^2/2, the moment dictionary:
S_mom = {X, Y, Px, Py, X^2, Y^2, Px^2, Py^2, sym(X Px), sym(Y Py)}
is exactly closed under commutation with H.

Commutation relations for H = (Px^2 + Py^2)/2:
  [H, X] = -i Px
  [H, Y] = -i Py
  [H, Px] = 0
  [H, Py] = 0
  [H, X^2] = -i (X Px + Px X) = -i sym(X Px)
  [H, Y^2] = -i (Y Py + Py Y) = -i sym(Y Py)
  [H, Px^2] = 0
  [H, Py^2] = 0
  [H, sym(X Px)] = -i (Px^2 + Px^2) = -2i Px^2
  [H, sym(Y Py)] = -i (Py^2 + Py^2) = -2i Py^2

So H_S can be constructed analytically!
"""
import numpy as np
from numpy import pi
from scipy import linalg


def build_position_operator(N, axis='x'):
    """
    Build position operator X or Y as diagonal matrix.

    Grid: x, y in [-pi, pi) with N points.

    Args:
        N: Grid size
        axis: 'x' or 'y'

    Returns:
        Diagonal matrix (N*N, N*N)
    """
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    if axis == 'x':
        vals = X.flatten()
    else:
        vals = Y.flatten()

    return np.diag(vals)


def build_momentum_operator(N, axis='x'):
    """
    Build momentum operator Px or Py.

    In position basis, P = -i d/dx, implemented via FFT:
    P |ψ⟩ = F^{-1} k F |ψ⟩

    Note: We use the convention that matches circuit_2D.py where
    k = fftfreq(N) * N gives k = 0, 1, ..., N/2-1, -N/2, ..., -1

    Args:
        N: Grid size
        axis: 'x' or 'y'

    Returns:
        Matrix (N*N, N*N)
    """
    # Build matrix by applying to each basis vector
    P = np.zeros((N*N, N*N), dtype='complex128')

    # Momentum values matching circuit_2D.py convention
    kvals = np.fft.fftfreq(N) * N

    for j in range(N*N):
        e_j = np.zeros(N*N, dtype='complex128')
        e_j[j] = 1.0
        psi = e_j.reshape(N, N)

        # Apply P via FFT
        psi_k = np.fft.fft2(psi)

        if axis == 'x':
            # Px: multiply by kx in Fourier space
            # kx varies along axis 1 (columns)
            KX = np.tile(kvals, (N, 1))  # shape (N, N), kx along columns
            P_psi_k = KX * psi_k
        else:
            # Py: multiply by ky in Fourier space
            # ky varies along axis 0 (rows)
            KY = np.tile(kvals.reshape(-1, 1), (1, N))  # shape (N, N), ky along rows
            P_psi_k = KY * psi_k

        P_psi = np.fft.ifft2(P_psi_k)
        P[:, j] = P_psi.flatten()

    return P


def build_kinetic_hamiltonian_from_momentum(Px, Py):
    """
    Build H = (Px^2 + Py^2) / 2.
    """
    return 0.5 * (Px @ Px + Py @ Py)


def symmetrize(A, B):
    """Compute symmetrized product sym(AB) = (AB + BA) / 2."""
    return 0.5 * (A @ B + B @ A)


def build_moment_observables(N):
    """
    Build the moment observable dictionary.

    S_mom = {X, Y, Px, Py, X^2, Y^2, Px^2, Py^2, sym(X Px), sym(Y Py)}

    Returns:
        observables: List of 10 matrices
        names: List of observable names
    """
    print("    Building X, Y operators...")
    X_op = build_position_operator(N, 'x')
    Y_op = build_position_operator(N, 'y')

    print("    Building Px, Py operators...")
    Px_op = build_momentum_operator(N, 'x')
    Py_op = build_momentum_operator(N, 'y')

    print("    Building quadratic operators...")
    X2_op = X_op @ X_op
    Y2_op = Y_op @ Y_op
    Px2_op = Px_op @ Px_op
    Py2_op = Py_op @ Py_op

    print("    Building symmetrized operators...")
    XPx_sym = symmetrize(X_op, Px_op)
    YPy_sym = symmetrize(Y_op, Py_op)

    observables = [X_op, Y_op, Px_op, Py_op, X2_op, Y2_op, Px2_op, Py2_op, XPx_sym, YPy_sym]
    names = ['X', 'Y', 'Px', 'Py', 'X^2', 'Y^2', 'Px^2', 'Py^2', 'sym(XPx)', 'sym(YPy)']

    return observables, names, {'X': X_op, 'Y': Y_op, 'Px': Px_op, 'Py': Py_op}


def build_shadow_hamiltonian_analytic():
    """
    Build H_S analytically from known commutation relations.

    Observable ordering: [X, Y, Px, Py, X^2, Y^2, Px^2, Py^2, sym(XPx), sym(YPy)]
    Indices:              0   1   2    3    4     5     6      7        8          9

    Commutation relations for H = (Px^2 + Py^2)/2:
      [H, X] = -i Px           -> h[0,2] = 1
      [H, Y] = -i Py           -> h[1,3] = 1
      [H, Px] = 0
      [H, Py] = 0
      [H, X^2] = -i sym(XPx)   -> h[4,8] = 1
      [H, Y^2] = -i sym(YPy)   -> h[5,9] = 1
      [H, Px^2] = 0
      [H, Py^2] = 0
      [H, sym(XPx)] = -2i Px^2 -> h[8,6] = 2
      [H, sym(YPy)] = -2i Py^2 -> h[9,7] = 2

    So [H, O_m] = -i ∑_{m'} h_{mm'} O_{m'}
    means d/dt <O_m> = -i [H, O_m] = -∑_{m'} h_{mm'} <O_{m'}>

    Wait, let me be more careful with signs.
    Heisenberg equation: d/dt O = i[H, O]
    So d/dt <O> = i <[H, O]>

    If [H, O_m] = -i ∑_{m'} h_{mm'} O_{m'}
    Then d/dt <O_m> = i <[H, O_m]> = i * (-i) ∑_{m'} h_{mm'} <O_{m'}> = ∑_{m'} h_{mm'} <O_{m'}>

    So o_dot = H_S @ o (real dynamics, not -i H_S)

    Let's verify: [H, X] = [(Px^2+Py^2)/2, X] = [Px^2, X]/2 = (Px [Px, X] + [Px, X] Px)/2
    [Px, X] = -i (in units where hbar=1)
    So [Px^2, X] = Px(-i) + (-i)Px = -2i Px
    [H, X] = -i Px

    d/dt <X> = i <[H, X]> = i * (-i) <Px> = <Px>

    So h[0, 2] = 1 means d/dt <X> = 1 * <Px> ✓

    Returns:
        H_S: (10, 10) real matrix
    """
    H_S = np.zeros((10, 10), dtype='float64')

    # d/dt <X> = <Px>
    H_S[0, 2] = 1.0

    # d/dt <Y> = <Py>
    H_S[1, 3] = 1.0

    # d/dt <Px> = 0 (free particle)
    # d/dt <Py> = 0

    # d/dt <X^2> = <sym(XPx)>
    H_S[4, 8] = 1.0

    # d/dt <Y^2> = <sym(YPy)>
    H_S[5, 9] = 1.0

    # d/dt <Px^2> = 0
    # d/dt <Py^2> = 0

    # d/dt <sym(XPx)> = 2 <Px^2>
    H_S[8, 6] = 2.0

    # d/dt <sym(YPy)> = 2 <Py^2>
    H_S[9, 7] = 2.0

    return H_S


def build_shadow_hamiltonian_from_commutators(H, observables, ridge_lambda=1e-10):
    """
    Build H_S by computing commutators and solving least squares.

    For each O_m: [H, O_m] = ? and we fit -i ∑_{m'} h_{mm'} O_{m'}

    Since d/dt <O_m> = i <[H, O_m]>, we have:
    If [H, O_m] = ∑_{m'} c_{mm'} O_{m'}, then d/dt <O_m> = i ∑_{m'} c_{mm'} <O_{m'}>

    Returns:
        H_S: Matrix such that o_dot = H_S @ o
        residuals: Closure residuals
    """
    M = len(observables)
    n = H.shape[0]

    # Vectorize observables
    O_vecs = np.zeros((M, n*n), dtype='complex128')
    for m, O_m in enumerate(observables):
        O_vecs[m, :] = O_m.flatten()

    H_S = np.zeros((M, M), dtype='complex128')
    residuals = []

    for m, O_m in enumerate(observables):
        # Compute [H, O_m]
        comm = H @ O_m - O_m @ H
        comm_vec = comm.flatten()

        # Solve: ∑_{m'} c_{mm'} O_{m'} ≈ [H, O_m]
        # Least squares: c_m = (O_vecs @ O_vecs.H)^{-1} @ O_vecs @ comm_vec
        A_mat = O_vecs.conj() @ O_vecs.T
        b_vec = O_vecs.conj() @ comm_vec

        reg = ridge_lambda * np.eye(M)
        c_m = np.linalg.solve(A_mat + reg, b_vec)

        # d/dt <O_m> = i <[H, O_m]> = i ∑_{m'} c_{mm'} <O_{m'}>
        # So H_S[m, :] = i * c_m
        H_S[m, :] = 1j * c_m

        # Residual
        reconstructed = sum(c_m[mp] * observables[mp] for mp in range(M))
        residual = np.linalg.norm(comm - reconstructed, 'fro')
        residuals.append(residual)

    return H_S, residuals


def compute_expectation(O, psi):
    """Compute <psi|O|psi>."""
    return np.vdot(psi, O @ psi)


def extract_moment_expectations(observables, psi):
    """Extract all moment expectations."""
    return np.array([compute_expectation(O, psi) for O in observables])


def evolve_shadow_moments(H_S, o0, dt, n_steps):
    """
    Evolve moment expectations: o_dot = H_S @ o

    Solution: o(t) = expm(H_S * t) @ o(0)
    """
    o_list = [o0.copy()]

    # Propagator for one time step
    propagator = linalg.expm(H_S * dt)

    o_current = o0.copy()
    for _ in range(n_steps):
        o_current = propagator @ o_current
        o_list.append(o_current.copy())

    return o_list


def reconstruct_gaussian_density(moments, N):
    """
    Reconstruct Gaussian density from moments.

    For a Gaussian, the density is:
    rho(x,y) = 1/(2*pi*sqrt(det(Sigma))) * exp(-1/2 * (r-mu)^T Sigma^{-1} (r-mu))

    where mu = (<X>, <Y>) and Sigma is the covariance matrix.

    Sigma = [[<X^2> - <X>^2, <XY> - <X><Y>],
             [<XY> - <X><Y>, <Y^2> - <Y>^2]]

    Note: We don't have <XY> in our observable set, so we assume <XY> = <X><Y>
    (uncorrelated, which is true for our initial state).

    Args:
        moments: [<X>, <Y>, <Px>, <Py>, <X^2>, <Y^2>, <Px^2>, <Py^2>, <sym(XPx)>, <sym(YPy)>]
        N: Grid size

    Returns:
        rho: (N, N) density array
    """
    # Extract moments (take real parts for physical quantities)
    mean_x = np.real(moments[0])
    mean_y = np.real(moments[1])
    var_x = np.real(moments[4]) - mean_x**2  # <X^2> - <X>^2
    var_y = np.real(moments[5]) - mean_y**2  # <Y^2> - <Y>^2

    # Ensure positive variance
    var_x = max(var_x, 1e-10)
    var_y = max(var_y, 1e-10)

    # Grid
    x = np.linspace(-pi, pi, N, endpoint=False)
    y = np.linspace(-pi, pi, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Gaussian density (assuming uncorrelated)
    rho = np.exp(-0.5 * ((X - mean_x)**2 / var_x + (Y - mean_y)**2 / var_y))
    rho = rho / (2 * pi * np.sqrt(var_x * var_y))

    # Normalize to sum to 1
    rho = rho / np.sum(rho)

    return rho


def extract_fourier_block(rho, K):
    """
    Extract low-frequency Fourier block from density.

    Returns:
        block: (2K+1, 2K+1) complex array
        energy: sum(|block|^2)
    """
    N = rho.shape[0]
    rho_hat = np.fft.fft2(rho)
    rho_hat_c = np.fft.fftshift(rho_hat)

    center = N // 2
    block = rho_hat_c[center-K:center+K+1, center-K:center+K+1]
    energy = np.sum(np.abs(block)**2)

    return block, energy
