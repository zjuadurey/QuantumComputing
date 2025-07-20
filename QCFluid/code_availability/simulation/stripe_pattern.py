from . import *
from math import pi

# 构建量子电路
def build_circuit_stripe_pattern(error_q_idx=6) -> simulator.Circuit:
    # 加载初态制备+时间演化到t=pi_over_2的qasm电路
    # diverging_flow_t=pi_over_2_test_4.qasm
    with open(root.joinpath('raw_circuit/diverging_flow_t=pi_over_2_test_5.qasm'),
              'r') as f:
        qasm = f.read()
    print("sssss")
    # 按照换行符分割
    qasm = qasm.split(';\n')
    # 用模拟器构建量子电路
    circuit = simulator.Circuit()
    for gate_info in qasm[3:-1]:
        _gate, _q = gate_info.split(' ')
        _q = _q.split(',')
        # 单比特门电路
        if len(_q) == 1:
            q_idx = int(_q[0][2:-1])
            if _gate.startswith('u'):
                theta, phi, angle_lambda = [
                    eval(angle)
                    for angle in re.findall('u\((.*),(.*),(.*)\)', _gate)[0]
                ]
                circuit.qiskit_u(q_idx, theta, phi, angle_lambda)
            else:
                raise NotImplementedError(_gate)
            # 引入人工误差
            if q_idx == error_q_idx:
                circuit.Rx(error_q_idx, 0.025)
        # 两比特门电路
        elif len(_q) == 2:
            q0_idx, q1_idx = _q
            q0_idx, q1_idx = int(q0_idx[2:-1]), int(q1_idx[2:-1])
            if _gate == 'cz':
                circuit.CZ([q0_idx, q1_idx])
            else:
                raise NotImplementedError(_gate)
        else:
            raise Exception(f'unknown gate {_gate} for {gate_info}')

    return circuit

# 模拟并可视化
def stripe_pattern_simulation(error_q_idx=6, save=False):
    print("stripe_pattern_simulation", error_q_idx, save)
    N = 2**5
    # 加载分组后的泡利串信息
    sampling_op_info = load_sampling_op_info()
    # 全局测量基集合
    sampling_op = sampling_op_info['sampling_op']
    # 构建量子电路
    c0 = build_circuit_stripe_pattern(error_q_idx)
    # 计算量子态在测量基'ZZZZZZZZZZ'下的概率分布
    probs = {'ZZZZZZZZZZ': c0.state_vector().probs(list(range(10)))}
    # 遍历全局测量基
    for _sampling_op in sampling_op:
        _c = c0.copy()
        # 逐个比特基底转换
        for _idx, _op in enumerate(_sampling_op):
            # 把 X 基变为 Z 基

            # 意义：Z基测量结果为 |0⟩ 或 |1⟩，好处理
            # 先作用 H，把X的本征态变为Z本征态，最后测量时等价于直接测X基
            if _op == 'X':
                _c.plus_gate(_idx, '-Y/2')
            # 把 Y 基变为 Z 基(近似)
            elif _op == 'Y':
                _c.plus_gate(_idx, 'X/2')
        probs[_sampling_op] = _c.state_vector().probs(list(range(10)))
    # 采样结果转换为物理量
    rho0, current, expectation = process_sampling_result(
        probs, collect_expectation=True)


    # 存储数据到excel
    if save:
        _x = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _y = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _xs, _ys = np.meshgrid(_x, _y)
        with pd.ExcelWriter(
                root.joinpath(
                    f"temp/stripe_pattern_q{error_q_idx}_error.xlsx")) as ew:
            pd.DataFrame(_xs).to_excel(ew, sheet_name='x')
            pd.DataFrame(_ys).to_excel(ew, sheet_name='y')
            pd.DataFrame(rho0).to_excel(ew, sheet_name='density')
            pd.DataFrame(current.real).to_excel(ew, sheet_name='momentum_x')
            pd.DataFrame(current.imag).to_excel(ew, sheet_name='momentum_y')
            pd.DataFrame({
                'expectation': expectation
            }).to_excel(ew, sheet_name='expectation')
    # 可视化
    x = np.linspace(0, 2**5 - 1, 2**5)
    y = np.linspace(0, 2**5 - 1, 2**5)
    xticks = np.linspace(0, 2**5 - 1, 5)
    yticks = np.linspace(0, 2**5 - 1, 5)
    xticklabels = [r'-$\pi$', r'-$\pi/2$', 0, r'$\pi/2$', r'$\pi$']
    yticklabels = [r'-$\pi$', r'-$\pi/2$', 0, r'$\pi/2$', r'$\pi$']
    xs, ys = np.meshgrid(x, y)

    def set_ax(ax):
        ax.set_xlim(0, 2**5 - 1)
        ax.set_ylim(0, 2**5 - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig = plt.figure(figsize=[7, 4])
    ax = fig.add_subplot(1, 2, 1)
    # im = ax.imshow(rho0, origin='lower', cmap='Blues', vmin=0, vmax=0.0025)
    im = ax.imshow(rho0, origin='lower', cmap='Blues', vmin=0, vmax=0.004)
    plt.colorbar(im, ax=ax, location='bottom')
    _title = 'density'
    ax.set_title(_title)
    set_ax(ax)

    ax = fig.add_subplot(1, 2, 2)
    # im = ax.imshow(np.abs(current),
    #                cmap='Blues',
    #                origin='lower',
    #                vmin=0,
    #                vmax=0.001)
    im = ax.imshow(np.abs(current),
                   cmap='Greens',
                   origin='lower',
                   vmin=0,
                   vmax=0.005)
    strm = ax.streamplot(xs,
                         ys,
                         current.real,
                         current.imag,
                         density=1,
                         color=np.abs(current),
                         cmap='Greys',
                         linewidth=0.5)
    # strm.lines.set_clim(0, 0.0005)
    strm.lines.set_clim(0, 0.0025)
    plt.colorbar(im, ax=ax, location='bottom')
    _title = 'momentum'
    ax.set_title(_title)
    set_ax(ax)

    plt.tight_layout()
