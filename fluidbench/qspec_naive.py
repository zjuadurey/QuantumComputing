"""
Naïve quantum-spectral solver: builds QFT-Phase-IQFT circuit and runs on Aer statevector.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from stateprep import GRID_POW

N   = 2 ** GRID_POW
NX  = NY = GRID_POW        # 5 qubits per axis → 32×32

class QuantumSpectralSolver:
    def __init__(self):
        self.backend = AerSimulator(method="statevector")

    # ------- circuit pieces -------
    @staticmethod
    def _kinetic(n, dt):
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

    def _step(self, statevec: Statevector, dt):
        qn = NX + NY + 1                     # +1 ancilla for two components
        qc = QuantumCircuit(qn)
        qc.initialize(statevec.data)
        qftx, qfty = QFT(NX), QFT(NY)
        iqftx, iqfty = qftx.inverse(), qfty.inverse()
        qc.append(qftx, range(NX))
        qc.append(qfty, range(NX, NX+NY))
        qc.append(self._kinetic(NX, dt), range(NX))
        qc.append(self._kinetic(NY, dt), range(NX, NX+NY))
        qc.append(iqftx, range(NX))
        qc.append(iqfty, range(NX, NX+NY))
        qc.save_state()

        job = self.backend.run(transpile(qc, self.backend, optimization_level=0))
        return Statevector(job.result().get_statevector(qc))

    # ------- public evolve -------
    def evolve(self, psi1_0, psi2_0, t_list):
        out = {}
        vec  = np.concatenate([psi1_0.reshape(-1), psi2_0.reshape(-1)])
        vec /= np.linalg.norm(vec)                 # ★ 关键归一化 ★
        state = Statevector(vec)

        last_t = t_list[0]
        out[last_t] = (psi1_0.copy(), psi2_0.copy())

        for t in t_list[1:]:
            dt = t - last_t
            state = self._step(state, dt)
            psi1, psi2 = state.data.reshape(2, N, N)
            out[t] = (psi1.copy(), psi2.copy())
            last_t = t
        return out
