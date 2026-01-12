//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include "include/Tensor.h"

static std::vector<std::size_t> shape1(std::size_t a){
    std::vector<std::size_t> s; s.push_back(a); return s;
}
static std::vector<std::size_t> shape2(std::size_t a, std::size_t b){
    std::vector<std::size_t> s; s.push_back(a); s.push_back(b); return s;
}

static std::vector<double> values_range(double start, double end_exclusive){
    std::vector<double> v;
    for (long long x = (long long)start; x < (long long)end_exclusive; ++x) v.push_back((double)x);
    return v;
}

int main() {
    std::srand(123);

    // 1) zeros / ones
    Tensor A = Tensor::zeros(shape2(2,3));
    Tensor B = Tensor::ones(shape2(2,3));
    std::cout << "A zeros 2x3:\n"; A.imprimir();
    std::cout << "B ones 2x3:\n";  B.imprimir();

    // 2) arange
    Tensor V = Tensor::arange(0, 6); // shape 6
    std::cout << "V arange 0..5:\n"; V.imprimir();

    // 3) random
    Tensor R = Tensor::random(shape2(2,3), -1.0, 1.0);
    std::cout << "R random 2x3 [-1,1):\n"; R.imprimir();

    // 4) suma normal
    Tensor C = A + B;
    std::cout << "C = A + B:\n"; C.imprimir();

    // 5) bias broadcast: (2x3) + (1x3)
    std::vector<double> bias_vals;
    bias_vals.push_back(10); bias_vals.push_back(20); bias_vals.push_back(30);
    Tensor bias(shape2(1,3), bias_vals);

    Tensor D = B + bias;
    std::cout << "D = B(2x3) + bias(1x3):\n"; D.imprimir();
    // esperado: cada fila de B (todo 1) suma 10,20,30

    // 6) escalar
    Tensor E = D * 2.0;
    std::cout << "E = D * 2:\n"; E.imprimir();

    std::cout << "OK\n";
    return 0;

}
