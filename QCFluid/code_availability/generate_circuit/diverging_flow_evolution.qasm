OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1,q2,q3,q4 { h q4; cp(pi/2) q4,q3; cp(pi/4) q4,q2; cp(pi/8) q4,q1; cp(pi/16) q4,q0; h q3; cp(pi/2) q3,q2; cp(pi/4) q3,q1; cp(pi/8) q3,q0; h q2; cp(pi/2) q2,q1; cp(pi/4) q2,q0; h q1; cp(pi/2) q1,q0; h q0; swap q0,q4; swap q1,q3; }
gate gate_QFT_133238548172224 q0,q1,q2,q3,q4 { gate_QFT q0,q1,q2,q3,q4; }
gate gate_IQFT_dg q0,q1,q2,q3,q4 { swap q1,q3; swap q0,q4; h q0; cp(-pi/2) q1,q0; h q1; cp(-pi/4) q2,q0; cp(-pi/2) q2,q1; h q2; cp(-pi/8) q3,q0; cp(-pi/4) q3,q1; cp(-pi/2) q3,q2; h q3; cp(-pi/16) q4,q0; cp(-pi/8) q4,q1; cp(-pi/4) q4,q2; cp(-pi/2) q4,q3; h q4; }
gate gate_IQFT q0,q1,q2,q3,q4 { gate_IQFT_dg q0,q1,q2,q3,q4; }
gate gate_QFT_133237416123936 q0,q1,q2,q3,q4 { gate_QFT q0,q1,q2,q3,q4; }
qreg q[10];
gate_QFT_133238548172224 q[0],q[1],q[2],q[3],q[4];
rz(-64*pi) q[0];
rz(-16*pi) q[1];
rz(-4*pi) q[2];
rz(-pi) q[3];
rz(-pi/4) q[4];
gate_IQFT q[0],q[1],q[2],q[3],q[4];
gate_QFT_133237416123936 q[5],q[6],q[7],q[8],q[9];
rz(-64*pi) q[5];
rz(-16*pi) q[6];
rz(-4*pi) q[7];
rz(-pi) q[8];
rz(-pi/4) q[9];
gate_IQFT q[5],q[6],q[7],q[8],q[9];