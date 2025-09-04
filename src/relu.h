// relu.h
#pragma once

#include "openfhe.h"
#include <vector>
#include <iostream>

using namespace lbcrypto;

// ================== Approximate ReLU Variants ==================
Ciphertext<DCRTPoly> ApproxReLU4_linear(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_x);
// 단순 x^2
Ciphertext<DCRTPoly> ApproxReLU4_square(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_x);

// CryptoNet 논문 기반 근사식 (0.25 + 0.5x + 0.125x^2)
Ciphertext<DCRTPoly> ApproxReLU4_cryptonet(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_x);

// 기존 quad_v3 (4차 다항식)
Ciphertext<DCRTPoly> ApproxReLU4_quad(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& ct_x);


// Studen ReLU
Ciphertext<DCRTPoly> ApproxReLU4_Student(
    CryptoContext<DCRTPoly> cc,
     const Ciphertext<DCRTPoly>& ct_x);

std::vector<Ciphertext<DCRTPoly>> ApplyApproxReLU4_All(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ct_channels,
    int mode = 2);

Ciphertext<DCRTPoly> ApproxReLU_Advanced(CryptoContext<DCRTPoly> cc, 
    const Ciphertext<DCRTPoly>& ct_x,
    double B);

Ciphertext<DCRTPoly> ApproxSign(CryptoContext<DCRTPoly> cc, 
    const Ciphertext<DCRTPoly>& ct_x, 
    double B);

Ciphertext<DCRTPoly> EvalPolynomial(CryptoContext<DCRTPoly> cc, 
    const Ciphertext<DCRTPoly>& ct_x,
    const std::vector<double>& coefficients);