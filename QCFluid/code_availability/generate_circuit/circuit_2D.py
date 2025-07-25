#%%
# =======================Generate state preparation quantum circuit===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.quantum_info import Operator
from numpy import pi, exp, sqrt, abs, real, imag
from scipy import linalg

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

def evolve(nx, ny, dt, initial_state):
    q_num = nx + ny + 1
    circ = QuantumCircuit(q_num)

    circ.initialize(initial_state) 
    circ.barrier()

    from qiskit.circuit.library import QFT
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

    # H = np.diag(np.fft.fftfreq(N)*N)
    # U = linalg.expm(-1j*H**2*dt/2)
    # circ.append(Operator(U), range(nx))
    # circ.append(Operator(U), range(nx, nx+ny))
    circ.barrier()

    circ.append(IQFT_x, range(nx))
    circ.append(IQFT_y, range(nx, nx+ny))

    circ.draw('mpl')

    circ.save_state()
    simulator = AerSimulator(method='statevector')
    circ = transpile(circ, simulator)

    result = simulator.run(circ).result()
    result.data(0)
    tmp = np.zeros(2**q_num, dtype='complex128')
    for i in range(2**q_num):
        tmp[i] = result.data(0)['statevector'][i]
    return tmp


# %%
# =========================Output the wave function==========================
def compute_fluid_quantities(psi1, psi2):
    kx = np.fft.fftfreq(N)*N
    ky = np.fft.fftfreq(N)*N
    KX, KY = np.meshgrid(kx, ky)
    psi1_spec = np.fft.fft2(psi1)
    dpsi1_x = np.fft.ifft2(1j*KX*psi1_spec)
    dpsi1_y = np.fft.ifft2(1j*KY*psi1_spec)
    psi2_spec = np.fft.fft2(psi2)
    dpsi2_x = np.fft.ifft2(1j*KX*psi2_spec)
    dpsi2_y = np.fft.ifft2(1j*KY*psi2_spec)
    rho = np.abs(psi1)**2 + np.abs(psi2)**2
    ux = real(real(psi1)*imag(dpsi1_x) - imag(psi1)*real(dpsi1_x) + real(psi2)*imag(dpsi2_x) - imag(psi2)*real(dpsi2_x)) / rho
    uy = real(real(psi1)*imag(dpsi1_y) - imag(psi1)*real(dpsi1_y) + real(psi2)*imag(dpsi2_y) - imag(psi2)*real(dpsi2_y)) / rho
    vor = real(np.fft.ifft2(1j*KX*np.fft.fft2(uy) - 1j*KY*np.fft.fft2(ux)))
    return ux, uy, vor

N = 2**5
x = np.linspace(-pi, pi, N, endpoint=False)
y = np.linspace(-pi, pi, N, endpoint=False)
dx = 2*pi/N
dy = 2*pi/N
X, Y = np.meshgrid(x, y)
R = sqrt(X**2 + Y**2)
sigma = 3
f = exp(-(R/sigma)**4)
u = 2*(X + 1j*Y)*f / (1 + R**2)
v = 1j*(R**2 + 1 - 2*f) / (1 + R**2)
psi1_0 = u / sqrt(abs(u)**2 + abs(v)**4)
psi2_0 = v**2 / sqrt(abs(u)**2 + abs(v)**4)

initial_state = np.array([psi1_0, psi2_0])
initial_state = initial_state.reshape(-1)
magnitude = np.linalg.norm(initial_state)
initial_state = initial_state/magnitude


# %%
# =========================Run the circuit==========================
nx = 5
ny = 5
dt = 1  # time step
tmp = evolve(nx, ny, dt, initial_state)

tmp = tmp.reshape(2, N, N)
psi1 = tmp[0, :, :]
psi2 = tmp[1, :, :]

ux, uy, vor = compute_fluid_quantities(psi1, psi2)

x = np.linspace(0, 2*pi, N, endpoint=False)
y = np.linspace(0, 2*pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)


fig_width = 8/2.54
fig_height = 8/2.54
fig = plt.figure(figsize=(fig_width, fig_height))

ax_width = 4/2.54 / fig_width
ax_height = 4/2.54 / fig_height
ax = fig.add_axes([0, 0, ax_width, ax_height])
ax.set_box_aspect(1/1)
ax.pcolormesh(X, Y, vor, cmap='Blues')