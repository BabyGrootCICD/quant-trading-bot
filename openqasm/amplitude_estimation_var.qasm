OPENQASM 3.0;
include "stdgates.inc";

gate log_normal_state_prep(q, mu, sigma, bounds_low, bounds_high) {
    // Prepare log-normal distribution for amplitude estimation
    // q[0]: sign qubit, q[1..n]: amplitude qubits
    h q[0];
    for i in [1: 3] {
        ry(2 * asin(sqrt((mu - bounds_low) / (bounds_high - bounds_low)))) q[i];
    }
    // Entangling for correlation structure
    for i in [1: 2] {
        cx q[i], q[i+1];
    }
}

gate amplitude_estimation_grover(q, a, k) {
    // Grover operator for amplitude estimation
    // A: state preparation
    // S0: reflection about |0>
    // S1: reflection about marked states
    // Q = -A S0 A^\dagger S1
    
    // Apply A
    log_normal_state_prep(q, 0.001, 0.02, 0.0, 0.005);
    
    // S1: phase flip on marked states (terminal value > strike)
    for i in [1: 3] {
        z q[i];
    }
    
    // A^\dagger
    // (inverse of state prep)
    for i in [2: 1] {
        cx q[i], q[i+1];
    }
    for i in [3: 1] {
        ry(-2 * asin(sqrt((0.001 - 0.0) / (0.005 - 0.0)))) q[i];
    }
    h q[0];
    
    // S0
    for i in [0: 3] {
        x q[i];
    }
    h q[3];
    mcx q[0], q[1], q[2], q[3];
    h q[3];
    for i in [0: 3] {
        x q[i];
    }
    
    // A
    log_normal_state_prep(q, 0.001, 0.02, 0.0, 0.005);
}

// Amplitude estimation for VaR/CVaR
// 4 qubits: 1 sign + 3 amplitude + 5 eval qubits
qubit[9] q;
bit[9] c;

const int n_eval = 5;

// Initialize eval register in |+>
for i in [4: 8] {
    h q[i];
}

// Apply controlled Grover operators
for j in [0: n_eval-1] {
    for rep in [0: (1 << j) - 1] {
        amplitude_estimation_grover(q[0:3], q[4+j], j);
    }
}

// Inverse QFT on eval register
for i in [4: 8] {
    for j in [4: i-1] {
        cp(-pi / 2^(i-j)) q[j], q[i];
    }
    h q[i];
}

// Swap for bit order
for i in [4: 6] {
    swap q[i], q[8 - (i - 4)];
}

// Measure eval register
for i in [4: 8] {
    c[i] = measure q[i];
}

// Also measure asset qubits
for i in [0: 3] {
    c[i] = measure q[i];
}