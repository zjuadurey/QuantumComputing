# Combine all essential components of the Shadow Hamiltonian Simulation code into one script

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- Parameters ---
N = 32
dx = 2 * np.pi / N
dy = 2 * np.pi / N
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
y = np.linspace(-np.pi, np.pi, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# --- Vortex field construction (Madelung-style) ---
x0, y0 = 0, 0
sigma = 3
R = np.sqrt((X - x0)**2 + (Y - y0)**2)
f = np.exp(-(R / sigma)**4)
u = 2 * (X + 1j * Y) * f / (1 + R**2)
v = 1j * (R**2 + 1 - 2 * f) / (1 + R**2)
psi1 = u / np.sqrt(np.abs(u)**2 + np.abs(v)**4)
psi2 = v**2 / np.sqrt(np.abs(u)**2 + np.abs(v)**4)

# --- Compute physical quantities: rho, Jx, Jy ---
kx = np.fft.fftfreq(N) * N
ky = np.fft.fftfreq(N) * N
KX, KY = np.meshgrid(kx, ky)

psi1_spec = np.fft.fft2(psi1)
dpsi1_x = np.fft.ifft2(1j * KX * psi1_spec)
dpsi1_y = np.fft.ifft2(1j * KY * psi1_spec)

psi2_spec = np.fft.fft2(psi2)
dpsi2_x = np.fft.ifft2(1j * KX * psi2_spec)
dpsi2_y = np.fft.ifft2(1j * KY * psi2_spec)

rho = np.abs(psi1)**2 + np.abs(psi2)**2
ux = np.real(np.real(psi1) * np.imag(dpsi1_x) - np.imag(psi1) * np.real(dpsi1_x)
             + np.real(psi2) * np.imag(dpsi2_x) - np.imag(psi2) * np.real(dpsi2_x)) / rho
uy = np.real(np.real(psi1) * np.imag(dpsi1_y) - np.imag(psi1) * np.real(dpsi1_y)
             + np.real(psi2) * np.imag(dpsi2_y) - np.imag(psi2) * np.real(dpsi2_y)) / rho

# --- Construct initial shadow state vector ---
rho_vec = rho.flatten()
ux_vec = ux.flatten()
uy_vec = uy.flatten()
shadow_state = np.concatenate([rho_vec, ux_vec, uy_vec])
shadow_state /= np.linalg.norm(shadow_state)

# --- Construct shadow Hamiltonian Hs ---
num_points = N * N
num_ops = 3 * num_points
data, rows, cols = [], [], []

def idx(x, y):
    return (x % N) + (y % N) * N

for y in range(N):
    for x in range(N):
        i = idx(x, y)
        i_rho = i
        i_jx = i + num_points
        i_jy = i + 2 * num_points

        for dx_shift, coeff in [(-1, 1), (1, -1)]:
            j = idx(x + dx_shift, y)
            j_jx = j + num_points
            data.append(coeff / (2 * dx))
            rows.append(i_rho)
            cols.append(j_jx)

        for dy_shift, coeff in [(-1, 1), (1, -1)]:
            j = idx(x, y + dy_shift)
            j_jy = j + 2 * num_points
            data.append(coeff / (2 * dy))
            rows.append(i_rho)
            cols.append(j_jy)

        for shift, coeff in [((0, 0), -4), ((1, 0), 1), ((-1, 0), 1), ((0, 1), 1), ((0, -1), 1)]:
            j = idx(x + shift[0], y + shift[1])
            j_rho = j
            data.append(coeff / dx**2)
            rows.append(i_jx)
            cols.append(j_rho)
            data.append(coeff / dy**2)
            rows.append(i_jy)
            cols.append(j_rho)

Hs = sp.csr_matrix((data, (rows, cols)), shape=(num_ops, num_ops))

# --- Time evolution ---
dt = 1.0
U = spla.expm(-1j * Hs * dt)
shadow_state_t1 = U @ shadow_state
shadow_rho_t1 = shadow_state_t1[:num_points].reshape(N, N).real

# --- Plot evolved density field ---
plt.figure(figsize=(5, 4))
plt.pcolormesh(X, Y, shadow_rho_t1, shading='auto', cmap='viridis')
plt.colorbar(label='Evolved ρ')
plt.title('Shadow Hamiltonian Simulation: ρ at t=1')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
