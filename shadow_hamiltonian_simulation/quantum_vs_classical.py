
import numpy as np
import pandas as pd
from numpy import pi, exp, sqrt, abs, real, imag
from scipy.fft import fft2, ifft2
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT

np.random.seed(42)
random.seed(42)

N = 32
nx = 5
ny = 5
qnum = nx + ny + 1
x = np.linspace(-pi, pi, N, endpoint=False)
y = np.linspace(-pi, pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)
timesteps = [1, 2, 3]
sigma_list = [pi / 8, pi / 6, pi / 4]
num_centers = 4
center_list = [(random.uniform(-pi, pi), random.uniform(-pi, pi)) for _ in range(num_centers)]

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

def evolve_quantum(nx, ny, dt, initial_state):
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
    statevector = result.data(0)['statevector']
    return np.array(statevector)

def generate_initial_psi(x0, y0, sigma):
    R = sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    f = exp(-(R / sigma) ** 4)
    u = 2 * ((X - x0) + 1j * (Y - y0)) * f / (1 + R ** 2)
    v = 1j * (R ** 2 + 1 - 2 * f) / (1 + R ** 2)
    psi1 = u / sqrt(abs(u) ** 2 + abs(v) ** 4)
    psi2 = v ** 2 / sqrt(abs(u) ** 2 + abs(v) ** 4)
    return psi1, psi2

def compute_rho_omega(psi1, psi2):
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    rho = abs(psi1) ** 2 + abs(psi2) ** 2
    dpsi1_x = ifft2(1j * KX * fft2(psi1))
    dpsi1_y = ifft2(1j * KY * fft2(psi1))
    dpsi2_x = ifft2(1j * KX * fft2(psi2))
    dpsi2_y = ifft2(1j * KY * fft2(psi2))
    ux = real(real(psi1) * imag(dpsi1_x) - imag(psi1) * real(dpsi1_x) +
              real(psi2) * imag(dpsi2_x) - imag(psi2) * real(dpsi2_x)) / rho
    uy = real(real(psi1) * imag(dpsi1_y) - imag(psi1) * real(dpsi1_y) +
              real(psi2) * imag(dpsi2_y) - imag(psi2) * real(dpsi2_y)) / rho
    omega = real(ifft2(1j * KX * fft2(uy) - 1j * KY * fft2(ux)))
    return rho, omega

def evolve_spectral(psi1_0, psi2_0, dt):
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    U = np.exp(-1j * K2 * dt / 2)
    psi1 = ifft2(fft2(psi1_0) * U)
    psi2 = ifft2(fft2(psi2_0) * U)
    return psi1, psi2

def compute_fidelity(psi1a, psi2a, psi1b, psi2b):
    a = psi1a + psi2a
    b = psi1b + psi2b
    a = a.flatten() / np.linalg.norm(a)
    b = b.flatten() / np.linalg.norm(b)
    return np.abs(np.vdot(a, b)) ** 2

def compute_L2_error(psi1a, psi1b):
    return np.linalg.norm(psi1a - psi1b) / np.linalg.norm(psi1b)

results = []
for (x0, y0) in center_list:
    for sigma in sigma_list:
        psi1_0, psi2_0 = generate_initial_psi(x0, y0, sigma)
        initial_state = np.array([psi1_0, psi2_0]).reshape(-1)
        initial_state /= np.linalg.norm(initial_state)

        for t in timesteps:
            psi1_classic, psi2_classic = evolve_spectral(psi1_0, psi2_0, dt=t)
            rho_c, omega_c = compute_rho_omega(psi1_classic, psi2_classic)
            mass_c = np.sum(rho_c)
            omega_max_c = np.max(np.abs(omega_c))
            statevector = evolve_quantum(nx, ny, dt=t, initial_state=initial_state)
            tmp = statevector.reshape(2, N, N)
            psi1_quantum = tmp[0]
            psi2_quantum = tmp[1]
            fidelity = compute_fidelity(psi1_quantum, psi2_quantum, psi1_classic, psi2_classic)
            error_psi1 = compute_L2_error(psi1_quantum, psi1_classic)
            error_psi2 = compute_L2_error(psi2_quantum, psi2_classic)
            results.append({
                'x0': x0,
                'y0': y0,
                'sigma': sigma,
                't': t,
                'mass_classic': mass_c,
                'omega_max_classic': omega_max_c,
                'fidelity': fidelity,
                'L2_psi1': error_psi1,
                'L2_psi2': error_psi2
            })

df = pd.DataFrame(results)
df.to_csv("quantum_vs_classical_metrics.csv", index=False)
print(df)
