# === quantum_evolve.py ===
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT

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
    from qiskit.quantum_info import Statevector
    q_num = nx + ny + 1
    circ = QuantumCircuit(q_num)
    circ.initialize(initial_state, range(q_num))
    circ.barrier()
    circ.append(QFT(nx), range(nx))
    circ.append(QFT(ny), range(nx, nx+ny))
    circ.barrier()
    circ.append(kinetic_operator(nx, dt), range(nx))
    circ.append(kinetic_operator(ny, dt), range(nx, nx+ny))
    circ.barrier()
    circ.append(QFT(nx).inverse(), range(nx))
    circ.append(QFT(ny).inverse(), range(nx, nx+ny))
    circ.save_statevector()

    simulator = AerSimulator(method="statevector")
    circ = transpile(circ, simulator)
    result = simulator.run(circ).result()
    statevector = result.data(0)['statevector']
    tmp = np.array(statevector).reshape(2, 2**nx, 2**ny)
    return tmp[0], tmp[1]