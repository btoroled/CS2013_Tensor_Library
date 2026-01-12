//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#include <iostream>
#include "include/Tensor.h"


int main() {

    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});

    std::cout << "dims=" << a.dims() << " numel=" << a.numel() << "\n";

    std::cout << "a(1,2)=" << a.at(1, 2) << "\n";
    a.imprimir();

    a.at(0, 0) = 99;
    a.imprimir();
    std::cout << "a(0,0) Despues de actualizarlo=" << a.at(0, 0) << "\n";

    return 0;
}