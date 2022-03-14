from process import Transition, Process, SUCCESS, FAIL
from dist_system import DistSystem, BlackboxEnvironment
from utils import *

OUTPUT_FILE = "execution_report.txt"


# permitted scenario
def permitted_experiment_setup():
    # transitions - system
    t1 = Transition('a', 'g1', 'g1')
    t2 = Transition('b', 'g1', 'g1')
    t3 = Transition('c', 'g1', 'g1')

    # transitions - environment
    t4 = Transition('a', 'e1', 'e2')
    t5 = Transition('b', 'e2', 'e3')
    t6 = Transition('c', 'e3', 'e1')

    # processes
    system = Process('sys', states=['g1'], transitions=[t1, t2, t3], initial_state='g1')
    environment = Process('env', states=['e1', 'e2', 'e3'], transitions=[t4, t5, t6], initial_state='e1')

    # distributed system
    dist_sys = BlackboxEnvironment('permitted', system=system, environment=environment)

    return dist_sys


# schedule scenario
def schedule_experiment_setup():
    # transitions - system
    t1 = Transition('a', 'g1', 'g2')
    t2 = Transition('b', 'g2', 'g1')
    t3 = Transition('b', 'g1', 'g3')
    t4 = Transition('c', 'g3', 'g1')

    # transitions - environment
    t5 = Transition('b', 'e1', 'e2')
    t6 = Transition('c', 'e2', 'e1')
    t7 = Transition('a', 'e1', 'e3')
    t8 = Transition('c', 'e3', 'e4')
    t9 = Transition('a', 'e4', 'e3')

    # processes
    system = Process('sys', states=['g1', 'g2', 'g3'], transitions=[t1, t2, t3, t4], initial_state='g1')
    environment = Process('env', states=['e1', 'e2', 'e3', 'e4'], transitions=[t5, t6, t7, t8, t9], initial_state='e1')

    # distributed system
    dist_sys = BlackboxEnvironment('schedule', system=system, environment=environment)

    return dist_sys


# cases scenario with an RNN controller
def cases_experiment_setup(history_len=1):
    # transitions - system
    t1 = Transition('a', 'g1', 'g2')
    t2 = Transition('b', 'g2', 'g1')
    t3 = Transition('c', 'g2', 'g1')

    # transitions - environment
    t4 = Transition('b', 'e1', 'e2')
    t5 = Transition('c', 'e1', 'e3')
    t6 = Transition('a', 'e2', 'e4')
    t7 = Transition('b', 'e4', 'e2')
    t8 = Transition('c', 'e3', 'e5')
    t9 = Transition('a', 'e5', 'e3')

    # processes
    system = Process('sys', states=['g1', 'g2'], transitions=[t1, t2, t3], initial_state='g1')
    environment = Process('env', states=['e1', 'e2', 'e3', 'e4', 'e5'], transitions=[t4, t5, t6, t7, t8, t9], initial_state='e1')

    # distributed system
    dist_sys = BlackboxEnvironment('cases', system=system, environment=environment, history_len=history_len)

    return dist_sys


# another cases scenario
def cases2_experiment_setup(history_len=1):
    # transitions - system
    t0 = Transition('a', 's0', 's1')
    t1 = Transition('b', 's1', 's2')
    t2 = Transition('c', 's1', 's2')
    t3 = Transition('d', 's1', 's2')
    t4 = Transition('b', 's2', 's0')
    t5 = Transition('c', 's2', 's0')
    t6 = Transition('d', 's2', 's0')

    # transitions - environment

    t7 = Transition('b', 'e0', 'e1')
    t8 = Transition('c', 'e0', 'e2')
    t9 = Transition('d', 'e0', 'e3')
    t10 = Transition('a', 'e1', 'e4')
    t11 = Transition('d', 'e4', 'e5')
    t12 = Transition('d', 'e5', 'e1')
    t13 = Transition('c', 'e2', 'e6')
    t14 = Transition('a', 'e6', 'e7')
    t15 = Transition('b', 'e7', 'e2')
    t16 = Transition('c', 'e3', 'e8')
    t17 = Transition('b', 'e8', 'e9')
    t18 = Transition('a', 'e9', 'e3')

    # processes
    system = Process('sys', states=['s0', 's1', 's2'], transitions=[t0, t1, t2, t3, t4, t5, t6], initial_state='s0')
    environment = Process('env',
                          states=['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9'],
                          transitions=[t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18], initial_state='e0')

    # distributed system
    dist_sys = BlackboxEnvironment('cases2', system=system, environment=environment, history_len=history_len)

    return dist_sys


# choice-scc scenario
def choice_scc_experiment_setup():
    # transitions - system
    t1 = Transition('a', 'g1', 'g2')
    t2 = Transition('b', 'g1', 'g2')
    t3 = Transition('c', 'g1', 'g2')
    t4 = Transition('d', 'g1', 'g2')
    t5 = Transition('e', 'g2', 'g2')

    # transitions - environment
    re1 = Transition('a', 'e0', 'a1')
    re2 = Transition('b', 'e0', 'b1')
    re3 = Transition('c', 'e0', 'c1')
    re4 = Transition('d', 'e0', 'd1')

    ra1 = Transition('e', 'a1', 'a2')
    ra2 = Transition('e', 'a2', 'a3')
    ra3 = Transition('e', 'a3', 'a4')
    ra4 = Transition('f', 'a4', 'a5')
    ra5 = Transition('f', 'a5', 'a6')
    ra6 = Transition('f', 'a6', 'a4')

    rb1 = Transition('e', 'b1', 'b2')
    rb2 = Transition('e', 'b2', 'b3')
    rb3 = Transition('f', 'b3', 'b4')
    rb4 = Transition('e', 'b4', 'b5')
    rb5 = Transition('f', 'b5', 'b6')
    rb6 = Transition('f', 'b6', 'b4')

    rc1 = Transition('e', 'c1', 'c2')
    rc2 = Transition('f', 'c2', 'c3')
    rc3 = Transition('f', 'c3', 'c4')
    rc4 = Transition('e', 'c4', 'c5')
    rc5 = Transition('e', 'c5', 'c6')
    rc6 = Transition('f', 'c6', 'c4')

    rd1 = Transition('f', 'd1', 'd2')
    rd2 = Transition('f', 'd2', 'd3')
    rd3 = Transition('f', 'd3', 'd4')
    rd4 = Transition('e', 'd4', 'd5')
    rd5 = Transition('e', 'd5', 'd6')
    rd6 = Transition('e', 'd6', 'd4')

    # processes
    system = Process('sys',
                     states=['g1', 'g2'],
                     transitions=[t1, t2, t3, t4, t5],
                     initial_state='g1')

    environment = Process('env',
                          states=['a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                                  'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                                  'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                                  'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
                                  'e0'],
                          transitions=[re1, re2, re3, re4,
                                       ra1, ra2, ra3, ra4, ra5, ra6,
                                       rb1, rb2, rb3, rb4, rb5, rb6,
                                       rc1, rc2, rc3, rc4, rc5, rc6,
                                       rd1, rd2, rd3, rd4, rd5, rd6],
                          initial_state='e0')

    # distributed system
    dist_sys = BlackboxEnvironment('choice_scc', system=system, environment=environment)

    return dist_sys


# cycle-scc scenario
def cycle_scc_experiment_setup(history_len):
    # transitions - system
    t1 = Transition('a', 'g1', 'g2')
    t2 = Transition('b', 'g1', 'g2')
    t3 = Transition('c', 'g1', 'g2')
    t4 = Transition('d', 'g1', 'g2')
    t5 = Transition('x', 'g2', 'g2')
    t6 = Transition('y', 'g2', 'g2')
    t7 = Transition('z', 'g2', 'g2')

    # transitions - environment
    re1 = Transition('a', 'e0', 'e1')
    re2 = Transition('b', 'e0', 'e7')
    re3 = Transition('c', 'e0', 'e13')
    re4 = Transition('d', 'e0', 'e19')

    ra1 = Transition('f', 'e1', 'e2')
    ra2 = Transition('f', 'e2', 'e3')
    ra3 = Transition('f', 'e3', 'e4')
    ra4 = Transition('x', 'e4', 'e5')
    ra5 = Transition('y', 'e5', 'e6')
    ra6 = Transition('z', 'e6', 'e4')

    rb1 = Transition('f', 'e7', 'e8')
    rb2 = Transition('f', 'e8', 'e9')
    rb3 = Transition('x', 'e9', 'e10')
    rb4 = Transition('y', 'e9', 'e10')
    rb5 = Transition('z', 'e9', 'e10')
    rb6 = Transition('f', 'e10', 'e11')
    rb7 = Transition('y', 'e11', 'e12')
    rb8 = Transition('z', 'e12', 'e10')

    rc1 = Transition('f', 'e13', 'e14')
    rc2 = Transition('x', 'e14', 'e15')
    rc3 = Transition('y', 'e14', 'e15')
    rc4 = Transition('z', 'e14', 'e15')
    rc5 = Transition('x', 'e15', 'e16')
    rc6 = Transition('y', 'e15', 'e16')
    rc7 = Transition('z', 'e15', 'e16')
    rc8 = Transition('f', 'e16', 'e17')
    rc9 = Transition('f', 'e17', 'e18')
    rc10 = Transition('z', 'e18', 'e16')

    rd1 = Transition('x', 'e19', 'e20')
    rd2 = Transition('y', 'e19', 'e20')
    rd3 = Transition('z', 'e19', 'e20')
    rd4 = Transition('x', 'e20', 'e21')
    rd5 = Transition('y', 'e20', 'e21')
    rd6 = Transition('z', 'e20', 'e21')
    rd7 = Transition('x', 'e21', 'e22')
    rd8 = Transition('y', 'e21', 'e22')
    rd9 = Transition('z', 'e21', 'e22')
    rd10 = Transition('f', 'e22', 'e23')
    rd11 = Transition('f', 'e23', 'e24')
    rd12 = Transition('f', 'e24', 'e22')

    # processes
    system = Process('sys',
                     states=['g1', 'g2'],
                     transitions=[t1, t2, t3, t4, t5, t6, t7],
                     initial_state='g1')

    environment = Process('env',
                          states=['e0',
                                  'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
                                  'e7', 'e8', 'e9', 'e10', 'e11', 'e12',
                                  'e13', 'e14', 'e15', 'e16', 'e17', 'e18',
                                  'e19', 'e20', 'e21', 'e22', 'e23', 'e24'],
                          transitions=[re1, re2, re3, re4,
                                       ra1, ra2, ra3, ra4, ra5, ra6,
                                       rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8,
                                       rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8, rc9, rc10,
                                       rd1, rd2, rd3, rd4, rd5, rd6, rd7, rd8, rd9, rd10, rd11, rd12],
                          initial_state='e0')

    # distributed system
    dist_sys = BlackboxEnvironment('cycle_scc', system=system, environment=environment, history_len=history_len)

    return dist_sys


# cycle-scc scenario
def cycle_scc_experiment_setup2():
    # transitions - system
    t01 = Transition('s', 'g01', 'g02')
    t02 = Transition('s', 'g02', 'g03')
    t03 = Transition('s', 'g03', 'g1')
    t1 = Transition('a', 'g1', 'g2')
    t2 = Transition('b', 'g1', 'g2')
    t3 = Transition('c', 'g1', 'g2')
    t4 = Transition('d', 'g1', 'g2')
    t5 = Transition('x', 'g2', 'g2')
    t6 = Transition('y', 'g2', 'g2')
    t7 = Transition('z', 'g2', 'g2')

    # transitions - environment
    rs1 = Transition('s', 'e01', 'e02')
    rs2 = Transition('s', 'e02', 'e03')
    rs3 = Transition('s', 'e03', 'e0')

    re1 = Transition('a', 'e0', 'e1')
    re2 = Transition('b', 'e0', 'e7')
    re3 = Transition('c', 'e0', 'e13')
    re4 = Transition('d', 'e0', 'e19')

    ra1 = Transition('f', 'e1', 'e2')
    ra2 = Transition('f', 'e2', 'e3')
    ra3 = Transition('f', 'e3', 'e4')
    ra4 = Transition('x', 'e4', 'e5')
    ra5 = Transition('y', 'e5', 'e6')
    ra6 = Transition('z', 'e6', 'e4')

    rb1 = Transition('f', 'e7', 'e8')
    rb2 = Transition('f', 'e8', 'e9')
    rb3 = Transition('x', 'e9', 'e10')
    rb4 = Transition('y', 'e9', 'e10')
    rb5 = Transition('z', 'e9', 'e10')
    rb6 = Transition('f', 'e10', 'e11')
    rb7 = Transition('y', 'e11', 'e12')
    rb8 = Transition('z', 'e12', 'e10')

    rc1 = Transition('f', 'e13', 'e14')
    rc2 = Transition('x', 'e14', 'e15')
    rc3 = Transition('y', 'e14', 'e15')
    rc4 = Transition('z', 'e14', 'e15')
    rc5 = Transition('x', 'e15', 'e16')
    rc6 = Transition('y', 'e15', 'e16')
    rc7 = Transition('z', 'e15', 'e16')
    rc8 = Transition('f', 'e16', 'e17')
    rc9 = Transition('f', 'e17', 'e18')
    rc10 = Transition('z', 'e18', 'e16')

    rd1 = Transition('x', 'e19', 'e20')
    rd2 = Transition('y', 'e19', 'e20')
    rd3 = Transition('z', 'e19', 'e20')
    rd4 = Transition('x', 'e20', 'e21')
    rd5 = Transition('y', 'e20', 'e21')
    rd6 = Transition('z', 'e20', 'e21')
    rd7 = Transition('x', 'e21', 'e22')
    rd8 = Transition('y', 'e21', 'e22')
    rd9 = Transition('z', 'e21', 'e22')
    rd10 = Transition('f', 'e22', 'e23')
    rd11 = Transition('f', 'e23', 'e24')
    rd12 = Transition('f', 'e24', 'e22')

    # processes
    system = Process('sys',
                     states=['g01', 'g02', 'g03', 'g1', 'g2'],
                     transitions=[t01, t02, t03, t1, t2, t3, t4, t5, t6, t7],
                     initial_state='g01')

    environment = Process('env',
                          states=['e01', 'e02', 'e03', 'e0',
                                  'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
                                  'e7', 'e8', 'e9', 'e10', 'e11', 'e12',
                                  'e13', 'e14', 'e15', 'e16', 'e17', 'e18',
                                  'e19', 'e20', 'e21', 'e22', 'e23', 'e24'],
                          transitions=[rs1, rs2, rs3, re1, re2, re3, re4,
                                       ra1, ra2, ra3, ra4, ra5, ra6,
                                       rb1, rb2, rb3, rb4, rb5, rb6, rb7, rb8,
                                       rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8, rc9, rc10,
                                       rd1, rd2, rd3, rd4, rd5, rd6, rd7, rd8, rd9, rd10, rd11, rd12],
                          initial_state='e01')

    # distributed system
    dist_sys = BlackboxEnvironment('cycle_scc', system=system, environment=environment)

    return dist_sys
