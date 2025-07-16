from calendar import c
from . import *
from math import pi

# 构建量子电路
def build_circuit_stripe_pattern(error_q_idx=6, circuit='vortex_circuit/vortex_phi_m_t=pi_over_2.qasm') -> simulator.Circuit:
    # 加载初态制备+时间演化到t=pi_over_2的qasm电路
    with open(root.joinpath(circuit),
              'r') as f:
        qasm = f.read()
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
            #if q_idx == error_q_idx:
            #    circuit.Rx(error_q_idx, 0.025)
        # 两比特门电路
        elif len(_q) == 2:
            q0_idx, q1_idx = _q
            q0_idx, q1_idx = int(q0_idx[2:-1]), int(q1_idx[2:-1])
            if _gate == 'cz':
                circuit.CZ([q0_idx, q1_idx])
            else:
                raise NotImplementedError(_gate)
        else:
            continue
            # print(f'barrier {_gate} for {gate_info}')
            # raise Exception(f'unknown gate {_gate} for {gate_info}')

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
    c0 = build_circuit_stripe_pattern(error_q_idx, 'vortex_circuit/vortex_phi_m_t=pi_over_2.qasm')
    c1 = build_circuit_stripe_pattern(error_q_idx, 'vortex_circuit/vortex_phi_p_t=pi_over_2.qasm')
    # 计算量子态在测量基'ZZZZZZZZZZ'下的概率分布
    probs0 = {'ZZZZZZZZZZ': c0.state_vector().probs(list(range(10)))}
    probs1 = {'ZZZZZZZZZZ': c1.state_vector().probs(list(range(10)))}
    # 遍历全局测量基
    # c0
    for _sampling_op in sampling_op:
        _c0 = c0.copy()
        # 逐个比特基底转换
        for _idx, _op in enumerate(_sampling_op):
            # 把 X 基变为 Z 基

            # 意义：Z基测量结果为 |0⟩ 或 |1⟩，好处理
            # 先作用 H，把X的本征态变为Z本征态，最后测量时等价于直接测X基
            if _op == 'X':
                _c0.plus_gate(_idx, '-Y/2')
            # 把 Y 基变为 Z 基(近似)
            elif _op == 'Y':
                _c0.plus_gate(_idx, 'X/2')
        probs0[_sampling_op] = _c0.state_vector().probs(list(range(10)))

    # c1
    for _sampling_op in sampling_op:
        _c1 = c1.copy()
        for _idx, _op in enumerate(_sampling_op):
            if _op == 'X':
                _c1.plus_gate(_idx, '-Y/2')
            elif _op == 'Y':
                _c1.plus_gate(_idx, 'X/2')
        probs1[_sampling_op] = _c1.state_vector().probs(list(range(10)))
    # 采样结果转换为物理量
    rho0, current0, expectation0 = process_sampling_result(
        probs0, collect_expectation=True)
    
    rho1, current1, expectation1 = process_sampling_result(
        probs1, collect_expectation=True)

    # 计算
    total_rho = rho0 + rho1
    total_current = current0 + current1

    def compute_vorticity(rho, current):
        if (rho.shape != (32, 32)):
            raise ValueError("Density matrix must be of shape (32, 32)")
        # 空间步长
        dx = dy = 2 * np.pi / 32

        # 速度场
        u_x = current.real / (rho + 1e-10)  # 避免除以0
        u_y = current.imag / (rho + 1e-10)

        # 初始化涡度
        vorticity = np.zeros((32, 32))

        # 中心差分法计算涡度（周期性边界）
        for i in range(32):
            for j in range(32):
                # ∂u_y/∂x (固定y，沿x方向差分)
                left_j = (j - 1) % 32
                right_j = (j + 1) % 32
                du_y_dx = (u_y[i, right_j] - u_y[i, left_j]) / (2 * dx)
                # ∂u_x/∂y (固定x，沿y方向差分)
                up_i = (i - 1) % 32
                down_i = (i + 1) % 32
                du_x_dy = (u_x[down_i, j] - u_x[up_i, j]) / (2 * dy)
                # 计算涡度
                vorticity[i, j] = du_y_dx - du_x_dy

        return vorticity
    
    vorticity = compute_vorticity(total_rho, total_current)

    # 存储数据到excel
    if save:
        _x = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _y = np.arange(-np.pi, np.pi, 2 * np.pi / 2**5)
        _xs, _ys = np.meshgrid(_x, _y)
        with pd.ExcelWriter(
                root.joinpath(
                    f"temp/vortex_stripe_pattern_q{error_q_idx}_error.xlsx")) as ew:
            pd.DataFrame(_xs).to_excel(ew, sheet_name='x')
            pd.DataFrame(_ys).to_excel(ew, sheet_name='y')
            pd.DataFrame(rho0 + rho1).to_excel(ew, sheet_name='density')
            pd.DataFrame(current0.real + current1.real).to_excel(ew, sheet_name='momentum_x')
            pd.DataFrame(current0.imag + current1.imag).to_excel(ew, sheet_name='momentum_y')
            pd.DataFrame({
                'expectation': expectation0
            }).to_excel(ew, sheet_name='expectation0')
            pd.DataFrame({
                'expectation': expectation1
            }).to_excel(ew, sheet_name='expectation1')
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

    fig = plt.figure(figsize=[10, 8])

    # 子图1: 总密度
    ax = fig.add_subplot(2, 2, 1)
    im = ax.imshow(total_rho, origin='lower', cmap='Blues')
    plt.colorbar(im, ax=ax, location='bottom')
    ax.set_title('Total Density')
    set_ax(ax)
    
    # 子图2: 总动量
    ax = fig.add_subplot(2, 2, 2)
    current_magnitude = np.abs(total_current)
    im = ax.imshow(current_magnitude, cmap='Greens', origin='lower')
    # 绘制流线图
    strm = ax.streamplot(xs, ys, total_current.real, total_current.imag,
                         density=1.5, color='k', linewidth=0.7)
    plt.colorbar(im, ax=ax, location='bottom')
    ax.set_title('Momentum Field')
    set_ax(ax)
    
    # 子图3: 涡度场
    ax = fig.add_subplot(2, 2, 3)
    # 使用红蓝双色表示正负涡度
    vort_max = np.max(np.abs(vorticity))
    im = ax.imshow(vorticity, origin='lower', cmap='coolwarm', 
                  vmin=-vort_max, vmax=vort_max)
    plt.colorbar(im, ax=ax, location='bottom')
    ax.set_title('Vorticity')
    set_ax(ax)
    
    # 子图4: 涡度幅度
    ax = fig.add_subplot(2, 2, 4)
    im = ax.imshow(np.abs(vorticity), origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, location='bottom')
    ax.set_title('Vorticity Magnitude')
    set_ax(ax)

    plt.tight_layout()
    plt.savefig(f"vortex_results_q{error_q_idx}.png", dpi=300)
    plt.show()
