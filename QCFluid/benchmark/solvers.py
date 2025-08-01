import numpy as np
from numpy import pi, real, imag

# ---------- 1. 经典谱一步演化 ----------
def run_spec_step(psi1, psi2, dt):
    N = psi1.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="xy")
    K2 = (KX**2 + KY**2) / 2.0
    phase = np.exp(-1j * K2 * dt)

    def evolve(comp):
        spec = np.fft.fft2(comp)
        spec *= phase
        return np.fft.ifft2(spec)

    return evolve(psi1), evolve(psi2)

# ---------- 2. 量子谱一步演化（可关闭） ----------
ENABLE_QSPEC = True
if ENABLE_QSPEC:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    def _kinetic_operator(n, dt):
        qc = QuantumCircuit(n)
        qc.rz(-2 ** (n - 1) * dt, n - 1)
        for i in range(n):
            qc.rz(2 ** (n - i - 2) * dt, n - i - 1)
        for i in range(1, n):
            qc.cx(n - 1, n - i - 1)
            qc.rz(-2 ** (2 * n - i - 2) * dt, n - i - 1)
            qc.cx(n - 1, n - i - 1)
        for i in range(n):
            for j in range(n):
                if i != j:
                    qc.cx(n - i - 1, n - j - 1)
                    qc.rz(2 ** (2 * n - i - j - 4) * dt, n - j - 1)
                    qc.cx(n - i - 1, n - j - 1)
        return qc

    def run_quantum_step(psi1, psi2, dt):
        N = psi1.shape[0]
        nx = ny = int(np.log2(N))
        qn = nx + ny + 1

        state_vec = np.vstack([psi1, psi2]).reshape(-1)
        state_vec /= np.linalg.norm(state_vec)

        from qiskit.circuit.library import QFT
        QFT_x, QFT_y = QFT(nx), QFT(ny)
        IQFT_x, IQFT_y = QFT_x.inverse(), QFT_y.inverse()

        circ = QuantumCircuit(qn)
        circ.initialize(state_vec)
        circ.append(QFT_x, range(nx))
        circ.append(QFT_y, range(nx, nx + ny))
        circ.append(_kinetic_operator(nx, dt), range(nx))
        circ.append(_kinetic_operator(ny, dt), range(nx, nx + ny))
        circ.append(IQFT_x, range(nx))
        circ.append(IQFT_y, range(nx, nx + ny))
        circ.save_state()

        sim = AerSimulator(method="statevector")
        circ = transpile(circ, sim)
        sv = np.asarray(sim.run(circ).result().data(0)["statevector"])

        psi1_new, psi2_new = sv.reshape(2, N, N)
        return psi1_new, psi2_new
else:
    run_quantum_step = None

# ---------- 3. ψ → ρ, jx, jy, vort ----------
def compute_fluid_quantities(psi1, psi2):
    N = psi1.shape[0]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="xy")

    rho = np.abs(psi1) ** 2 + np.abs(psi2) ** 2

    def grad(comp):
        spec = np.fft.fft2(comp)
        return (
            np.fft.ifft2(1j * KX * spec),
            np.fft.ifft2(1j * KY * spec),
        )

    d1x, d1y = grad(psi1)
    d2x, d2y = grad(psi2)

    ux = (real(psi1) * imag(d1x) - imag(psi1) * real(d1x)
          + real(psi2) * imag(d2x) - imag(psi2) * real(d2x)) / rho
    uy = (real(psi1) * imag(d1y) - imag(psi1) * real(d1y)
          + real(psi2) * imag(d2y) - imag(psi2) * real(d2y)) / rho

    vort = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(uy)
                                - 1j * KY * np.fft.fft2(ux)))

    return rho.astype(np.float32), (rho * ux).astype(np.float32), \
           (rho * uy).astype(np.float32), vort.astype(np.float32)
