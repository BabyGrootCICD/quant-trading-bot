OPENQASM 3.0;
include "stdgates.inc";

gate qaoa_cost_layer(q, gamma, mu, sigma, q_risk, budget) {
    // Cost Hamiltonian for portfolio optimization
    // Z_i Z_j terms from covariance matrix
    for i in [0: n-1] {
        for j in [i+1: n-1] {
            rzz(2 * gamma * q_risk * sigma[i][j]) q[i], q[j];
        }
    }
    // Z_i terms from expected returns
    for i in [0: n-1] {
        rz(-2 * gamma * mu[i]) q[i];
    }
    // Budget constraint penalty
    for i in [0: n-1] {
        rz(2 * gamma * q_risk * (2 * budget - n)) q[i];
    }
}

gate qaoa_mixer_layer(q, beta) {
    // X mixer
    for i in [0: n-1] {
        rx(2 * beta) q[i];
    }
}

// QAOA circuit for portfolio optimization
// n qubits = n assets, p layers
qubit[8] q;
bit[8] c;

const float mu[8] = {0.001, 0.002, 0.0015, 0.0008, 0.0012, 0.001, 0.0009, 0.0011};
const float sigma[8][8] = {
    {0.0004, 0.0001, 0.0001, 0.00005, 0.00008, 0.00006, 0.00004, 0.00005},
    {0.0001, 0.0005, 0.00015, 0.00006, 0.0001, 0.00008, 0.00005, 0.00006},
    {0.0001, 0.00015, 0.0003, 0.00004, 0.00007, 0.00005, 0.00003, 0.00004},
    {0.00005, 0.00006, 0.00004, 0.0002, 0.00003, 0.00002, 0.00001, 0.00002},
    {0.00008, 0.0001, 0.00007, 0.00003, 0.00025, 0.00004, 0.00002, 0.00003},
    {0.00006, 0.00008, 0.00005, 0.00002, 0.00004, 0.00015, 0.00002, 0.00002},
    {0.00004, 0.00005, 0.00003, 0.00001, 0.00002, 0.00002, 0.0001, 0.00001},
    {0.00005, 0.00006, 0.00004, 0.00002, 0.00003, 0.00002, 0.00001, 0.00012}
};
const float q_risk = 0.5;
const int budget = 4;
const int p = 3;

float[8] gamma;
float[8] beta;

for layer in [0: p-1] {
    gamma[layer] = 0.5 + layer * 0.1;
    beta[layer] = 0.3 + layer * 0.05;
}

for layer in [0: p-1] {
    qaoa_cost_layer(q, gamma[layer], mu, sigma, q_risk, budget);
    qaoa_mixer_layer(q, beta[layer]);
}

for i in [0: 7] {
    c[i] = measure q[i];
}