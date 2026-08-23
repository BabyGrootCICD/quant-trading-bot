OPENQASM 3.0;
include "stdgates.inc";

gate zz_feature_map(q, x, reps) {
    // ZZFeatureMap for quantum kernel
    // Encodes classical data x into quantum state
    for rep in [0: reps-1] {
        // Hadamard layer
        for i in [0: 7] {
            h q[i];
        }
        // Data encoding
        for i in [0: 7] {
            rz(2 * x[i]) q[i];
        }
        // Entangling layer
        for i in [0: 6] {
            for j in [i+1: 7] {
                rzz(2 * (pi - x[i]) * (pi - x[j])) q[i], q[j];
            }
        }
    }
}

gate swap_test(q_a, q_b, ancilla) {
    // SWAP test for fidelity = |<phi|psi>|^2
    h ancilla;
    for i in [0: 7] {
        cswap ancilla, q_a[i], q_b[i];
    }
    h ancilla;
}

// Quantum kernel circuit for SVM
// 8 feature qubits + 8 feature qubits + 1 ancilla = 17 qubits
qubit[17] q;
bit[1] c;

// Classical data vectors (would be loaded from data)
float[8] x1 = {0.1, 0.2, 0.15, 0.05, 0.12, 0.08, 0.03, 0.07};
float[8] x2 = {0.11, 0.18, 0.16, 0.04, 0.13, 0.09, 0.02, 0.06};

// Prepare first state |phi> = U(x1)|0>
zz_feature_map(q[0:7], x1, 2);

// Prepare second state |psi> = U(x2)|0>
zz_feature_map(q[8:15], x2, 2);

// SWAP test
swap_test(q[0:7], q[8:15], q[16]);

// Measure ancilla - probability of |0> = fidelity
c[0] = measure q[16];

// fidelity = P(ancilla = 0) = |<phi|psi>|^2