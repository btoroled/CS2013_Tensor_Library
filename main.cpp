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

int main() {
    // ----- VIEW -----
    Tensor A = Tensor::arange(0, 12);

    std::vector<std::size_t> s;
    s.push_back(3);
    s.push_back(4);

    Tensor B = A.view(s);

    std::cout << "B = view(3x4) de arange(0,12):\n";
    B.imprimir();

    // A quedó válido pero vacío (por move)
    std::cout << "A numel despues de view (deberia ser 0): " << A.numel() << "\n";

    // ----- UNSQUEEZE -----
    Tensor C = Tensor::arange(0, 3); // [0,1,2]

    Tensor D = C.unsqueeze(0); // {1,3}
    std::cout << "D = C.unsqueeze(0) (shape deberia ser 1x3):\n";
    D.imprimir();
    std::cout << "C numel despues de unsqueeze (deberia ser 0): " << C.numel() << "\n";

    Tensor E = Tensor::arange(0, 3);
    Tensor F = E.unsqueeze(1); // {3,1}
    std::cout << "F = E.unsqueeze(1) (shape deberia ser 3x1):\n";
    F.imprimir();

    return 0;
}