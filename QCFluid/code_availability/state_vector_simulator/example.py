from . import simulator

# GHZ 态是一种 最大纠缠的多量子比特态,所有量子比特要么同时处于0态要么同时处于1态
# 测量结果应集中在全 0 和全 1 的状态，其他状态的出现表明存在错误（如噪声或门误差）。
# 测试量子硬件的纠缠能力，验证量子模拟器的正确性
def GHZ_test(q_num=10):
    print('-' * 10)
    print(f'**  {q_num}q GHZ state **')
    print('-' * 10)
    c = simulator.Circuit()
    c.plus_gate(0, 'H')
    for i in range(1, q_num):
        c.CNOT([0, i])
    print('counts:')
    print(c.state_vector().counts())
