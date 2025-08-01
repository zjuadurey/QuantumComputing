
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT

# ========== Qiskit Quantum Evolution ==========
def kinetic_operator(n, dt):
    qc = QuantumCircuit(n)
    qc.rz(-2**(n-1)*dt, n-1)
    for i in range(n):
        qc.rz(2**(n-i-2)*dt, n-i-1)
    for i in range(1, n):
        qc.cx(n-1, n-i-1)
        qc.rz(-2**(2*n-i-2)*dt, n-i-1)
        qc.cx(n-1, n-i-1)
    for i in range(n):
        for j in range(n):
            if i != j:
                qc.cx(n-i-1, n-j-1)
                qc.rz(2**(2*n-i-j-4)*dt, n-j-1)
                qc.cx(n-i-1, n-j-1)
    return qc

def evolve_once_qiskit(nx, ny, dt, initial_state):
    q_num = nx + ny + 1
    circ = QuantumCircuit(q_num)
    circ.initialize(initial_state)
    circ.barrier()

    QFT_x = QFT(nx)
    QFT_y = QFT(ny)
    IQFT_x = QFT(nx).inverse()
    IQFT_y = QFT(ny).inverse()

    kinetic_x = kinetic_operator(nx, dt)
    kinetic_y = kinetic_operator(ny, dt)

    circ.append(QFT_x, range(nx))
    circ.append(QFT_y, range(nx, nx+ny))
    circ.barrier()

    circ.append(kinetic_x, range(nx))
    circ.append(kinetic_y, range(nx, nx+ny))
    circ.barrier()

    circ.append(IQFT_x, range(nx))
    circ.append(IQFT_y, range(nx, nx+ny))
    circ.save_state()

    simulator = AerSimulator(method='statevector')
    circ = transpile(circ, simulator)
    result = simulator.run(circ).result()
    vec = result.data(0)['statevector']
    return np.array(vec)

def evolve_qiskit_multi(initial_state, nx, ny, dt, steps):
    N = 2**nx
    psi1s, psi2s = [], []
    state = initial_state.copy()

    for _ in range(steps+1):
        state = state / np.linalg.norm(state)
        tmp = state.reshape(2, N, N)
        psi1s.append(tmp[0].copy())
        psi2s.append(tmp[1].copy())
        state = evolve_once_qiskit(nx, ny, dt, state)

    return psi1s, psi2s

# ========== Spectral Baseline ==========
def evolve_spectral(psi1_0, psi2_0, dt, steps, N):
    kx = fftfreq(N) * N
    ky = fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    U = np.exp(-1j * K2 * dt)

    psi1, psi2 = psi1_0.copy(), psi2_0.copy()
    psi1s, psi2s = [psi1.copy()], [psi2.copy()]

    for _ in range(steps):
        psi1_fft = fft2(psi1)
        psi2_fft = fft2(psi2)
        psi1 = ifft2(psi1_fft * U)
        psi2 = ifft2(psi2_fft * U)
        psi1s.append(psi1.copy())
        psi2s.append(psi2.copy())

    return psi1s, psi2s

# ========== Fluid Quantities ==========
def compute_fluid_quantities(psi1, psi2, N):
    kx = fftfreq(N) * N
    ky = fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    psi1_spec = fft2(psi1)
    psi2_spec = fft2(psi2)
    dpsi1_x = ifft2(1j*KX*psi1_spec)
    dpsi1_y = ifft2(1j*KY*psi1_spec)
    dpsi2_x = ifft2(1j*KX*psi2_spec)
    dpsi2_y = ifft2(1j*KY*psi2_spec)
    rho = np.abs(psi1)**2 + np.abs(psi2)**2
    ux = np.real(np.real(psi1)*np.imag(dpsi1_x) - np.imag(psi1)*np.real(dpsi1_x) +
                 np.real(psi2)*np.imag(dpsi2_x) - np.imag(psi2)*np.real(dpsi2_x)) / (rho + 1e-12)
    uy = np.real(np.real(psi1)*np.imag(dpsi1_y) - np.imag(psi1)*np.real(dpsi1_y) +
                 np.real(psi2)*np.imag(dpsi2_y) - np.imag(psi2)*np.real(dpsi2_y)) / (rho + 1e-12)
    vor = np.real(ifft2(1j*KX*fft2(uy) - 1j*KY*fft2(ux)))
    return ux, uy, vor

def compare_vorticities(vorticities_q, vorticities_c):
    l2_errors, correlations = [], []
    for vq, vc in zip(vorticities_q, vorticities_c):
        delta = vq - vc
        l2 = np.linalg.norm(delta) / np.linalg.norm(vq)
        corr = pearsonr(vq.flatten(), vc.flatten())[0]
        l2_errors.append(l2)
        correlations.append(corr)
    return l2_errors, correlations

def plot_errors(l2_errors, correlations, dt):
    times = np.arange(len(l2_errors)) * dt
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(times, l2_errors, label='L2 Error', color='tab:red')
    ax1.set_ylabel('Relative L2 Error', color='tab:red')
    ax2 = ax1.twinx()
    ax2.plot(times, correlations, label='Correlation', color='tab:blue')
    ax2.set_ylabel('Correlation Coefficient', color='tab:blue')
    ax1.set_xlabel('Time')
    fig.tight_layout()
    plt.title('Quantum vs Spectral Classical Evolution')
    plt.grid(True)
    plt.show()

# ========== Init ==========
from numpy import pi, exp, sqrt
N = 32
x = np.linspace(-pi, pi, N, endpoint=False)
y = np.linspace(-pi, pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
sigma = 3
f = exp(-(R/sigma)**4)
u = 2*(X + 1j*Y)*f / (1 + R**2)
v = 1j*(R**2 + 1 - 2*f) / (1 + R**2)
psi1_0 = u / sqrt(np.abs(u)**2 + np.abs(v)**4)
psi2_0 = v**2 / sqrt(np.abs(u)**2 + np.abs(v)**4)

initial_state = np.array([psi1_0, psi2_0]).reshape(-1)
initial_state /= np.linalg.norm(initial_state)

# ========== Run ==========
dt = 1.0
steps = 5
nx = ny = 5
psi1s_q, psi2s_q = evolve_qiskit_multi(initial_state, nx, ny, dt, steps)
psi1s_c, psi2s_c = evolve_spectral(psi1_0, psi2_0, dt, steps, N)

vorticities_q = [compute_fluid_quantities(p1, p2, N)[2] for p1, p2 in zip(psi1s_q, psi2s_q)]
vorticities_c = [compute_fluid_quantities(p1, p2, N)[2] for p1, p2 in zip(psi1s_c, psi2s_c)]

l2s, cors = compare_vorticities(vorticities_q, vorticities_c)
plot_errors(l2s, cors, dt)

# Export CSV
df = pd.DataFrame({'Time': np.arange(len(l2s)) * dt, 'L2_Error': l2s, 'Correlation': cors})
df.to_csv("quantum_vs_spectral_real.csv", index=False)
