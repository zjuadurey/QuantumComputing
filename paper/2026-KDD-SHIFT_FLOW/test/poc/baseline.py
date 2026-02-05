"""
Baseline quantum evolution using Qiskit.
Adapted from circuit_2D.py (vortex code).
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT


def kinetic_operator(n, dt):
    """Kinetic energy operator for n qubits."""
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
    """
    Evolve the initial state for time dt using quantum circuit simulation.

    Args:
        nx: Number of qubits for x dimension
        ny: Number of qubits for y dimension
        dt: Time step
        initial_state: Initial statevector (length 2^(nx+ny+1))

    Returns:
        Final statevector as numpy array
    """
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
    tmp = np.zeros(2**q_num, dtype='complex128')
    for i in range(2**q_num):
        tmp[i] = result.data(0)['statevector'][i]
    return tmp
