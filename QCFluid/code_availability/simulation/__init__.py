import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import pathlib
import sys
import math
import pickle
from functools import reduce
import tqdm

from state_vector_simulator import simulator
from . import decompose_matrix

root = pathlib.Path(__file__).parent
if not root.joinpath('temp').exists():
    root.joinpath('temp').mkdir()

# 将量子操作符字符串转换为正则表达式模式
# 用于匹配量子电路中的泡利字符串
def get_regex(op):
    pattern = []
    for i in op:
        # X/Y 直接保留
        if i in 'XY':
            pattern.append(i)
        elif i == 'Z':
            pattern.append('[IZ]')
        elif i == 'I':
            pattern.append('[IXYZ]')
        else:
            raise RuntimeError(op + ' exist unknown character ' + i)
    return re.compile(''.join(pattern))


# 构造二维离散格点系统中某点 (x,y) 上的动量算符的对称差分近似
# 用格点间的差分代替连续导数，使动量算符离散化
# 在量子格点模型中，初始动量的大小并非直接由经典速度或质量决定，而是通过离散化算符的结构隐含确定
def current_matrix(N, x, y, periodic=False):
    """
    x->column;y->row
    generate current_matrix in (x,y)

    N:qubit number in 1 dimention
    x:[0,2**N-1]
    y:[0,2**N-1]
    periodic: True if using periodic bond condition
    """
    
    jx = np.zeros((2**(2 * N), 2**(2 * N)), dtype=np.complex128)
    jy = np.zeros((2**(2 * N), 2**(2 * N)), dtype=np.complex128)
    delta = 2 * np.pi / 2**N # 离散化步长（无量纲）
    element = 1 / (4j * delta) # 
    if periodic:
        jy[y * 2**N + x][((y + 1) * 2**N + x) % (2**(2 * N))] = element
        jy[y * 2**N + x][((y - 1) * 2**N + x) % (2**(2 * N))] = -element
        jy[((y - 1) * 2**N + x) % (2**(2 * N))][y * 2**N + x] = element
        jy[((y + 1) * 2**N + x) % (2**(2 * N))][y * 2**N + x] = -element

        jx[y * 2**N + x][(y * 2**N + x + 1) % (2**(2 * N))] = element
        jx[y * 2**N + x][(y * 2**N + x - 1) % (2**(2 * N))] = -element
        jx[(y * 2**N + x - 1) % (2**(2 * N))][y * 2**N + x] = element
        jx[(y * 2**N + x + 1) % (2**(2 * N))][y * 2**N + x] = -element
    else:
        if y == 2**N - 1:
            jy[y * 2**N + x][(y - 1) * 2**N + x] = -element * 2
            jy[(y - 1) * 2**N + x][y * 2**N + x] = element * 2
        elif y == 0:
            jy[y * 2**N + x][(y + 1) * 2**N + x] = element * 2
            jy[(y + 1) * 2**N + x][y * 2**N + x] = -element * 2
        else:
            jy[y * 2**N + x][(y + 1) * 2**N + x] = element
            jy[(y + 1) * 2**N + x][y * 2**N + x] = -element
            jy[y * 2**N + x][(y - 1) * 2**N + x] = -element
            jy[(y - 1) * 2**N + x][y * 2**N + x] = element

        if x == 2**N - 1:
            jx[y * 2**N + x][y * 2**N + x - 1] = -element * 2
            jx[y * 2**N + x - 1][y * 2**N + x] = element * 2
        elif x == 0:
            jx[y * 2**N + x][y * 2**N + x + 1] = element * 2
            jx[y * 2**N + x + 1][y * 2**N + x] = -element * 2
        else:
            jx[y * 2**N + x][y * 2**N + x + 1] = element
            jx[y * 2**N + x + 1][y * 2**N + x] = -element
            jx[y * 2**N + x][y * 2**N + x - 1] = -element
            jx[y * 2**N + x - 1][y * 2**N + x] = element

    return jx, jy


# 把算符分解为泡利串，并去掉系数小于一定阈值的项
def dump_decomposed_current_matrix(threshold=1e-6):
    print(
        '[dump_decomposed_current_matrix]: Calculating... (This process may cost half an hour)'
    )
    N = 2**5
    jx_coeff = {}
    jy_coeff = {}
    for _x in tqdm.tqdm(range(N), desc='x'):
        for _y in tqdm.tqdm(range(N), desc='y'):
            jx, jy = current_matrix(5, _x, _y)
            _jx_coeff = decompose_matrix.decompose(
                jx)  # tr(j_x @ Pauli string)
            _jy_coeff = decompose_matrix.decompose(jy)
            jx_coeff[(_x, _y)] = {
                k: v / 2**10
                for k, v in _jx_coeff.items() if abs(v) > threshold
            }
            jy_coeff[(_x, _y)] = {
                k: v / 2**10
                for k, v in _jy_coeff.items() if abs(v) > threshold
            }
    with open(root.joinpath('temp/decomposed_current_matrix.pkl'), 'wb') as f:
        pickle.dump({'jx': jx_coeff, 'jy': jy_coeff}, f)


def load_decomposed_current_matrix():
    with open(root.joinpath('temp/decomposed_current_matrix.pkl'), 'rb') as f:
        result = pickle.load(f)
    return result


if not root.joinpath('temp/decomposed_current_matrix.pkl').exists():
    dump_decomposed_current_matrix()
DECOMPOSED_CURRENT_MATRIX = load_decomposed_current_matrix()

# 泡利分组测量预处理部分
# 对动量算符的泡利分解结果进行分析和分组
def dump_sampling_op_info():
    print('[dump_sampling_op_info]: Calculating...')
    result = DECOMPOSED_CURRENT_MATRIX
    jx = result['jx']
    jy = result['jy']
    # 收集两个方向的所有泡利串（去重）
    Pauli_string = {}
    for v in jx.values():
        Pauli_string.update(v)
    for v in jy.values():
        Pauli_string.update(v)
    Pauli_string = list(Pauli_string)
    # 根据泡利字符串中非 I 的比特数（即实际需要测量的比特数）分组
    sampling_op_full = {}
    for _s in Pauli_string:
        # 根据 _s 中 非 I 的比特数，将 _s 分组到 sampling_op_full 字典中
        # sampling_op_full.setdefault(key, [])：字典操作
        # 若 key（如 2）不存在于 sampling_op_full 中，则插入 key: []（空列表）
        # 若 key 已存在，直接返回对应的 value（列表）
        sampling_op_full.setdefault(len(_s) - _s.count('I'), []).append(_s)

    # found sampling op
    # 获取 sampling_op_full 的所有键
    op_size = list(sampling_op_full)
    # 初始化一个空字典 sampling_op，键为 op_size 中的值，值为空列表
    sampling_op = {_op_size: [] for _op_size in op_size}
    # 存储最终选出的全局测量基
    existed_sampling_op = []
    # 遍历分组并筛选测量基
    # 按非 I 比特数从大到小遍历（如先处理 3，再处理 2)，优先处理需要更多比特测量的字符串，提高复用率
    for _op_size in sorted(op_size, reverse=True):
        # 遍历_op_size相同一组中的所有泡利字符串
        for _sampling_op in sampling_op_full[_op_size]:
            # 正则表达式匹配
            # 举例：'XXI' → 生成匹配 'XXI'、'XXX'、'XXZ' 等的正则表达式（忽略 I 对应的比特）
            # 例如：有了'XXX'结果后，通过经典后处理，丢弃第二比特结果，就可以直接计算'XXI'
            # 计算 'XXX' 的具体含义是：通过实验测量和统计，确定量子态在全局泡利基 'XXX' 下的概率分布
            # 对第 3 个比特的所有可能值求和，得到前 2 个比特的联合概率，就是 'XXI' 的概率分布
            regex = get_regex(_sampling_op)
            found_sampling_op = False
            for _existed_sampling_op in existed_sampling_op:
                # 判断当前泡利串是否能被 existed_sampling_op 中已有的全局基覆盖
                _match = regex.match(_existed_sampling_op)
                if _match is not None:
                    found_sampling_op = True
                    break
            # 如果没有找到匹配的全局基，则将当前泡利串添加到全局基 existed_sampling_op 中
            if not found_sampling_op:
                sampling_op[_op_size].append(_sampling_op)
                existed_sampling_op.append(_sampling_op)

    # # found map from result to sampling op
    # 存储 局部泡利字符串 -> 对应的可复用全局测量基 的映射关系
    sampling_op_full_map = {}
    for _op_size in sorted(op_size, reverse=True):
        for _sampling_op in sampling_op_full[_op_size]:
            regex = get_regex(_sampling_op)
            # 遍历已存在的全局测量基，查找匹配的字符串
            # 例如：如果 'XXI' 可以被 'XXX' 覆盖，则将 'XXX' 添加到 sampling_op_full_map['XXI'] 中
            # 这样在处理 'XXI' 时，就可以直接查找 'XXX' 的结果并忽略最后一个比特
            for _existed_sampling_op in existed_sampling_op:
                _match = regex.match(_existed_sampling_op)
                if _match is not None:
                    sampling_op_full_map.setdefault(_sampling_op,
                                                    []).append(_match.group())
    with open(root.joinpath('temp/decomposed_current_matrix_info.pkl'),
              'wb') as f:
        pickle.dump(
            {
                'sampling_op_full_map': sampling_op_full_map,
                'sampling_op': existed_sampling_op
            }, f)


def load_sampling_op_info():
    with open(root.joinpath('temp/decomposed_current_matrix_info.pkl'),
              'rb') as f:
        result = pickle.load(f)
    return result


if not root.joinpath('temp/decomposed_current_matrix_info.pkl').exists():
    dump_sampling_op_info()
SAMPLING_OP_INFO = load_sampling_op_info()


# 将给定的概率分布转换为期望值
def prob_to_expect(probs: np.ndarray) -> float:
    '''probs: array'''
    qnum = int(math.log2(len(probs)))
    factors = reduce(np.outer, [np.array([1.0, -1.0])] * qnum).reshape(-1)
    result = np.sum(factors * probs)
    return result


def ptrace(probs, remain_q_idxs):
    '''
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

# 泡利分组测量
def process_sampling_result(probs: dict, collect_expectation=False):
    N = 2**5
    decomposed_current_matrix, sampling_op_info = DECOMPOSED_CURRENT_MATRIX, SAMPLING_OP_INFO
    sampling_op_full_map = sampling_op_info['sampling_op_full_map']
    # 全 Z 测量是获取密度矩阵对角元
    # 结果就是量子态的概率幅，根据马德龙变换，直接对应流体密度
    rho0 = probs['ZZZZZZZZZZ'].reshape(32, 32)

    current_x = np.zeros((N, N), dtype=np.complex128)
    current_y = np.zeros((N, N), dtype=np.complex128)
    _expectation_cache = {}
    # 遍历泡利串和其对应的可复用基底的集合
    for _Pauli_string, _target_ops in sampling_op_full_map.items():
        # 选择第一个可复用的泡利串基底
        _target_op = _target_ops[0]  # we choose the first one
        # 确定非I比特（确定保留哪些部分）
        remain_q_idxs = tuple(
            [idx for idx, _op in enumerate(_Pauli_string) if _op != 'I'])
        # 获得基底概率密度
        _probs = probs[_target_op]
        # 忽略 I 比特，对全局测量概率 _probs 求和
        _expectation_cache[_Pauli_string] = prob_to_expect(
            ptrace(_probs, remain_q_idxs))

    for x in tqdm.tqdm(range(N), desc='processing sampling result'):
        for y in range(N):
            #获取(x,y)坐标下两个方向的泡利分解
            _jx = decomposed_current_matrix['jx'][(x, y)]
            _jy = decomposed_current_matrix['jy'][(x, y)]
            # calculate <jx>
            _jx_expectation = 0
            for _Pauli_string, _coe in _jx.items():
                # 遇到单位矩阵I，直接加泡利串前面的系数
                if _Pauli_string == 'I' * len(_Pauli_string):
                    _jx_expectation += _coe

                else:
                    _jx_expectation += _coe * _expectation_cache[_Pauli_string]
            # calculate <jy>
            _jy_expectation = 0
            for _Pauli_string, _coe in _jy.items():
                if _Pauli_string == 'I' * len(_Pauli_string):
                    _jy_expectation += _coe
                else:
                    _jy_expectation += _coe * _expectation_cache[_Pauli_string]
            current_x[y][x] = _jx_expectation
            current_y[y][x] = _jy_expectation
    current_x = current_x.real
    current_y = current_y.real
    #复数存储两个方向的动量数值
    current = current_x + 1j * current_y

    if collect_expectation:
        return rho0, current, _expectation_cache
    else:
        return rho0, current
