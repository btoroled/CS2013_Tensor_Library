//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include "include/Tensor.h"

static std::vector<std::size_t> shape2(std::size_t a, std::size_t b) {
    std::vector<std::size_t> s;
    s.push_back(a);
    s.push_back(b);
    return s;
}

static std::vector<std::size_t> shape3(std::size_t a, std::size_t b, std::size_t c) {
    std::vector<std::size_t> s;
    s.push_back(a);
    s.push_back(b);
    s.push_back(c);
    return s;
}

static void print_shape(const Tensor& t, const char* name) {
    std::cout << name << " shape: (";
    const std::vector<std::size_t>& sh = t.shape();
    for (std::size_t i = 0; i < sh.size(); ++i) {
        std::cout << sh[i];
        if (i + 1 < sh.size()) std::cout << ", ";
    }
    std::cout << ")\n";
}

static void print_row_prefix(const Tensor& t, std::size_t row, std::size_t k, const char* name) {
    std::cout << name << " row " << row << " first " << k << " values: ";
    for (std::size_t j = 0; j < k; ++j) {
        std::cout << t.at(row, j) << " ";
    }
    std::cout << "\n";
}

int main() {
    std::srand(0);

    ReLU relu;
    Sigmoid sigmoid;

    //
    // PASO 1 Crear un tensor de entrada de dimensiones 1000 × 20 × 20.
    //

    Tensor X = Tensor::random(shape3(1000, 20, 20), 1, 20);

    //
    // PASO 2Transformarlo a 1000 ×400 usando view.
    //

    X = X.view(shape2(1000, 400));

    //
    // PASO 3 Multiplicarlo por una matriz 400 x 100
    //

    Tensor W1 = Tensor::random(shape2(400, 100), -0.5, 0.5);

    Tensor Z1 = matmul(X, W1);

    //
    //PASO 4 Sumar una matriz 1 x 100
    //
    Tensor b1 = Tensor::random(shape2(1, 100), -0.1, 0.1);

    Tensor Z1b = Z1 + b1;

    //
    // PASO 5 Aplicar la función ReLU.
    //
    Tensor A1 = Z1b.apply(relu);

    //
    // PASO 6 Multiplicar por una matriz 100 ×10
    //

    Tensor W2 = Tensor::random(shape2(100, 10), -0.5, 0.5);
    Tensor Z2 = matmul(A1, W2);

    //
    // PASO 7 Suma con bias b2 (1 ×10)
    //
    Tensor b2 = Tensor::random(shape2(1, 10), -0.1, 0.1);
    Tensor Z2b = Z2 + b2;

    //
    // PASO 8 Activación Sigmoid
    //
    Tensor Y = Z2b.apply(sigmoid);

    print_shape(X,  "X (flattened)");
    print_shape(W1, "W1");
    print_shape(b1, "b1");
    print_shape(A1, "A1 (after ReLU)");
    print_shape(W2, "W2");
    print_shape(b2, "b2");
    print_shape(Y,  "Y (output)");

    print_row_prefix(Y, 0, 10, "Y");

    return 0;

}


