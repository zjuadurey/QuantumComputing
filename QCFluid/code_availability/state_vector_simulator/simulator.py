from . import *
from . import Gate
from scipy.stats import unitary_group
from . import state_vector
from typing import Optional

# 对多量子比特的概率分布进行部分迹（Partial Trace）操作
# 部分迹：保留指定的量子比特（remain_q_idxs），并对其他比特进行求和约化
# 举例：
# 在量子计算中，计算泡利算符（如 X⊗I⊗Z）的期望值时，
# I（单位矩阵）对应的比特不需要测量
# 通过 ptrace 剔除无关比特，只保留需要测量的比特，再计算期望值
def ptrace(probs, remain_q_idxs):
    '''
    probs:测量后得到各种比特的概率分布
    ptrace prob with ptrace index
    probs.shape: (..., D)
    '''
    probs = np.asarray(probs)
    shape = list(probs.shape)
    D = shape[-1]
    _shape = shape[:-1]
    q_num = int(math.log2(D))
    return np.sum(
        np.moveaxis(probs.reshape(_shape + [2] * q_num),
                    np.asarray(remain_q_idxs) + len(_shape),
                    np.arange(len(remain_q_idxs)) +
                    len(_shape)).reshape(_shape + [2**len(remain_q_idxs), -1]),
        axis=-1)

# 将给定的概率分布转换为期望值

def prob_to_expect(probs: np.ndarray) -> float:
    qnum = int(math.log2(len(probs)))

    factors = reduce(np.outer, [np.array([1.0, -1.0])] * qnum).reshape(-1)
    # 期望值 = 本征值 * 概率
    result = np.sum(factors * probs)
    return result

# 继承自 state_vector.ProductStateVector
class Circuit(state_vector.ProductStateVector):
    # 计算泡利矩阵串的期望值
    # Pauli_str: 泡利算符字符串，如 'XXY' 表示
    # 第一个量子比特应用 X 门，第二个量子比特应用 X 门，第三个量子比特应用 Y 门
    def Pauli_expectation(self,
                          Pauli_str: str,
                          qubits: Optional[List[int]] = None) -> float: # 指定泡利算符作用的量子比特编号列表（默认作用于全部量子比特）
        all_qubits = sorted(reduce(lambda x, y: x + y, self.qubits))
        # 默认全部比特，否则报错
        if qubits is None:
            assert len(Pauli_str) == self.qnum
            qubits = all_qubits
        else:
            assert len(Pauli_str) == len(qubits)
        _copy = self.copy()
        remain_qubits = []
        # 处理每个量子比特的泡利算符
        # 实际量子计算机通常直接在 Z 基下测量，所以全部转到Z基底
        for _qubit, _Pauli_str in zip(qubits, Pauli_str):
            if _Pauli_str == 'I': # 单位矩阵，直接忽略
                continue
            elif _Pauli_str == 'X':
                # 先对量子比特施加 Ry(-π/2)，将 |+⟩ 和 |-⟩ 变为 |0⟩ 和 |1⟩，再测 Z
                _copy.Ry(_qubit, -np.pi / 2)
                remain_qubits.append(_qubit)
            elif _Pauli_str == 'Y':
                # 先施加 Rx(π/2)，将 |+i⟩ 和 |-i⟩ 近似变为 |0⟩ 和 |1⟩，再测 Z
                _copy.Rx(_qubit, np.pi / 2)
                remain_qubits.append(_qubit)
            elif _Pauli_str == 'Z':
                # 已经是Z基底，直接测量
                remain_qubits.append(_qubit)
            else: # 非法的泡利算符，报错
                raise ValueError(_Pauli_str)
        # 所有Pauli算符都是单位算符，期望值为1
        if len(remain_qubits) == 0:
            return 1.0
        # 计算概率分布
        probs = _copy.state_vector().probs(all_qubits)

        return prob_to_expect(
            ptrace(probs, [ # 部分迹，只保留需要测量的比特
                _idx for _idx, _qubit in enumerate(all_qubits)
                if _qubit in remain_qubits
            ]))

    def qiskit_u(self, qubit: int, theta, phi, angle_lambda):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.qiskit_u(theta, phi, angle_lambda))

    def rotation(self, qubit: int, alpha, theta, phi):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.rotation(alpha, theta, phi))

    def Rx(self, qubit: int, alpha):
        self.rotation(qubit, alpha, np.pi / 2, 0)

    def Ry(self, qubit: int, alpha):
        self.rotation(qubit, alpha, np.pi / 2, np.pi / 2)

    def Rz(self, qubit: int, alpha):
        self.rotation(qubit, alpha, 0, 0)

    def plus_gate(self, qubit: int, gate):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], getattr(Gate, Gate.gate_map.get(gate,
                                                                   gate)))

    def random_SU2(self, qubit: int):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], unitary_group.rvs(2))

    def Clifford1(self, qubit: int, Clifford_idx: int):
        assert isinstance(qubit, int)
        self.apply_matrix(qubit, Gate.Clifford1[Clifford_idx])

    def XEB_op(self, qubit: int, XEB_op_idx: int):
        assert isinstance(qubit, int)
        self.apply_matrix([qubit], Gate.XEBops[XEB_op_idx])

    def CZ(self, qubits: List[int]):
        assert isinstance(qubits, list)
        self.apply_matrix(qubits, Gate.CZ)

    def CNOT(self, qubits: List[int]):
        assert isinstance(qubits, list)
        self.apply_matrix(qubits, Gate.CNOT)

    # 噪声模拟
    # 单比特噪声（如 T1/T2）和 两比特噪声（如 CNOT 门的串扰）占据了量子系统错误的 99% 以上。

    # 单量子比特非对称去极化噪声
    # 以不同概率施加 X、Y、Z 错误
    def asymmetrical_depolarization_1q(self, qubit: int, p_X: float,
                                       p_Y: float, p_Z: float):
        assert isinstance(qubit, int)
        _r = np.random.rand()
        if _r < (1 - p_X - p_Y - p_Z):  # apply I
            pass
        elif _r < 1 - p_X - p_Y:  # apply Z
            self.apply_matrix([qubit], Gate.Z)
        elif _r < 1 - p_Y:  # apply X
            self.apply_matrix([qubit], Gate.X)
        else:  #apply Y
            self.apply_matrix([qubit], Gate.Y)
    # 单量子比特对称去极化噪声
    # 总错误概率均分给X/Y/Z门
    # 模拟各向同性退相干
    def depolarization_1q(self, qubit: int, p: float):
        self.asymmetrical_depolarization_1q(qubit, p / 3, p / 3, p / 3)
    # 相位阻尼噪声T2
    def phase_damping(self, qubit: int, gamma: float):
        '''
        Exponential-decay dephasing(T2), gamma=2*t_gate/T2
        Phase damping has exactly the same effect with phase flip.
        '''
        self.asymmetrical_depolarization_1q(qubit,
                                            p_X=0,
                                            p_Y=0,
                                            p_Z=(1 - (1 - gamma)**0.5) / 2)
    # 振幅阻尼噪声T1
    def amplitude_damping(self, qubit: int, gamma: float):
        '''
        Energy relaxation(T1), gamma=t_gate/T1
        '''
        _r = np.random.rand()
        _state_vector: state_vector.StateVector = self.state_vector([qubit])
        _state_vector.set_qubit_order([qubit])
        _state_tensor = _state_vector._state_tensor.reshape([2, -1])
        _P1 = np.sum(np.abs(_state_tensor[1])**2)
        if _r < gamma * _P1:
            _state_tensor[0] = _state_tensor[1] / _P1**0.5
            _state_tensor[1] = 0
        else:
            _state_tensor[1] *= (1 - gamma)**0.5
            _state_tensor /= (1 - _P1 * gamma)**0.5
        _state_vector._state_tensor = _state_tensor.reshape(
            _state_vector._shape)
    # 两量子比特非对称去极化噪声
    # 噪声（如串扰、耦合器失真）主要发生在存在直接相互作用的比特对之间
    def asymmetrical_depolarization_2q(self, qubits: List[int], p_IX: float,
                                       p_IY: float, p_IZ: float, p_XI: float,
                                       p_XX: float, p_XY: float, p_XZ: float,
                                       p_YI: float, p_YX: float, p_YY: float,
                                       p_YZ: float, p_ZI: float, p_ZX: float,
                                       p_ZY: float, p_ZZ: float):
        assert isinstance(qubits, list)
        _r = np.random.rand()
        if _r < (1 -
                 (p_IX + p_IY + p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI +
                  p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply II
            pass
        elif _r < (1 - (p_IY + p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX +
                        p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IX
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_IZ + p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY +
                        p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IY
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_XI + p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ +
                        p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply IZ
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (1 - (p_XX + p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI +
                        p_ZX + p_ZY + p_ZZ)):  # apply XI
            self.apply_matrix([qubits[0]], Gate.X)
        elif _r < (1 - (p_XY + p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX +
                        p_ZY + p_ZZ)):  # apply XX
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_XZ + p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY +
                        p_ZZ)):  # apply XY
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_YI + p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY +
                        p_ZZ)):  # apply XZ
            self.apply_matrix([qubits[0]], Gate.X)
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (
                1 -
            (p_YX + p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YI
            self.apply_matrix([qubits[0]], Gate.Y)
        elif _r < (1 - (p_YY + p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YX
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_YZ + p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YY
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.Y)
        elif _r < (1 - (p_ZI + p_ZX + p_ZY + p_ZZ)):  # apply YZ
            self.apply_matrix([qubits[0]], Gate.Y)
            self.apply_matrix([qubits[1]], Gate.Z)
        elif _r < (1 - (p_ZX + p_ZY + p_ZZ)):  # apply ZI
            self.apply_matrix([qubits[0]], Gate.Z)
        elif _r < (1 - (p_ZY + p_ZZ)):  # apply ZX
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.X)
        elif _r < (1 - (p_ZZ)):  # apply ZY
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.Y)
        else:  # apply ZZ
            self.apply_matrix([qubits[0]], Gate.Z)
            self.apply_matrix([qubits[1]], Gate.Z)
    # 两量子比特对称去极化噪声
    def depolarization_2q(self, qubits: List[int], p: float):
        self.asymmetrical_depolarization_2q(qubits, *([p / 15] * 15))
