//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include "include/Tensor.h"


// ------------------------
// Helpers de testing simples
// ------------------------
static int g_pass = 0;
static int g_fail = 0;

void CHECK(bool cond, const char* msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::cout << "[FAIL] " << msg << "\n";
    }
}

void CHECK_EQ_DOUBLE(double a, double b, const char* msg, double eps = 1e-12) {
    double diff = a - b;
    if (diff < 0) diff = -diff;
    CHECK(diff <= eps, msg);
}

std::vector<std::size_t> shape1(std::size_t a) {
    std::vector<std::size_t> s;
    s.push_back(a);
    return s;
}

std::vector<std::size_t> shape2(std::size_t a, std::size_t b) {
    std::vector<std::size_t> s;
    s.push_back(a);
    s.push_back(b);
    return s;
}

std::vector<std::size_t> shape3(std::size_t a, std::size_t b, std::size_t c) {
    std::vector<std::size_t> s;
    s.push_back(a);
    s.push_back(b);
    s.push_back(c);
    return s;
}

std::vector<double> linspace_values(std::size_t n, double start = 0.0) {
    std::vector<double> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) v.push_back(start + (double)i);
    return v;
}

// ------------------------
// Tests
// ------------------------
void test_constructor_and_at_1d() {
    std::vector<std::size_t> s = shape1(5);
    std::vector<double> v = linspace_values(5, 10.0); // 10,11,12,13,14

    Tensor t(s, v);

    CHECK(t.dims() == 1, "1D: dims debe ser 1");
    CHECK(t.numel() == 5, "1D: numel debe ser 5");
    CHECK_EQ_DOUBLE(t.at(0), 10.0, "1D: at(0) debe ser 10");
    CHECK_EQ_DOUBLE(t.at(4), 14.0, "1D: at(4) debe ser 14");

    // modificar y verificar
    t.at(2) = 99.0;
    CHECK_EQ_DOUBLE(t.at(2), 99.0, "1D: escritura/lectura en at(2)");
}

void test_constructor_and_at_2d_mapping() {
    // shape 2x3 => row-major:
    // [0,1,2,
    //  3,4,5]
    std::vector<std::size_t> s = shape2(2, 3);
    std::vector<double> v = linspace_values(6, 0.0);

    Tensor t(s, v);

    CHECK(t.dims() == 2, "2D: dims debe ser 2");
    CHECK(t.numel() == 6, "2D: numel debe ser 6");

    CHECK_EQ_DOUBLE(t.at(0,0), 0.0, "2D: at(0,0)=0");
    CHECK_EQ_DOUBLE(t.at(0,2), 2.0, "2D: at(0,2)=2");
    CHECK_EQ_DOUBLE(t.at(1,0), 3.0, "2D: at(1,0)=3");
    CHECK_EQ_DOUBLE(t.at(1,2), 5.0, "2D: at(1,2)=5");

    // escritura
    t.at(1,1) = -7.0;
    CHECK_EQ_DOUBLE(t.at(1,1), -7.0, "2D: escritura/lectura at(1,1)");
}

void test_constructor_and_at_3d_mapping() {
    // shape 2x2x3 (A=2,B=2,C=3), row-major
    // i=0:
    //   j=0: 0 1 2
    //   j=1: 3 4 5
    // i=1:
    //   j=0: 6 7 8
    //   j=1: 9 10 11
    std::vector<std::size_t> s = shape3(2, 2, 3);
    std::vector<double> v = linspace_values(12, 0.0);

    Tensor t(s, v);

    CHECK(t.dims() == 3, "3D: dims debe ser 3");
    CHECK(t.numel() == 12, "3D: numel debe ser 12");

    CHECK_EQ_DOUBLE(t.at(0,0,0), 0.0, "3D: at(0,0,0)=0");
    CHECK_EQ_DOUBLE(t.at(0,0,2), 2.0, "3D: at(0,0,2)=2");
    CHECK_EQ_DOUBLE(t.at(0,1,0), 3.0, "3D: at(0,1,0)=3");
    CHECK_EQ_DOUBLE(t.at(1,0,0), 6.0, "3D: at(1,0,0)=6");
    CHECK_EQ_DOUBLE(t.at(1,1,2), 11.0, "3D: at(1,1,2)=11");

    t.at(1,1,1) = 123.0;
    CHECK_EQ_DOUBLE(t.at(1,1,1), 123.0, "3D: escritura/lectura at(1,1,1)");
}

void test_copy_constructor_deep_copy() {
    std::vector<std::size_t> s = shape2(2, 2);
    std::vector<double> v = linspace_values(4, 1.0); // 1,2,3,4
    Tensor a(s, v);

    Tensor b(a); // copia

    a.at(0,0) = 999.0;
    CHECK_EQ_DOUBLE(b.at(0,0), 1.0, "Copy ctor: debe ser copia profunda (b no cambia)");
}

void test_copy_assignment_deep_copy() {
    std::vector<std::size_t> s1 = shape1(3);
    std::vector<double> v1 = linspace_values(3, 10.0);
    Tensor a(s1, v1);

    std::vector<std::size_t> s2 = shape2(2,2);
    std::vector<double> v2 = linspace_values(4, 0.0);
    Tensor b(s2, v2);

    b = a; // copy assignment

    CHECK(b.dims() == 1, "Copy assign: dims copiado");
    CHECK(b.numel() == 3, "Copy assign: numel copiado");
    CHECK_EQ_DOUBLE(b.at(2), 12.0, "Copy assign: valores copiados");

    a.at(1) = -5.0;
    CHECK_EQ_DOUBLE(b.at(1), 11.0, "Copy assign: copia profunda (b no cambia)");
}

void test_move_constructor() {
    std::vector<std::size_t> s = shape2(2,3);
    std::vector<double> v = linspace_values(6, 0.0);
    Tensor a(s, v);

    Tensor b(std::move(a));

    CHECK(b.dims() == 2, "Move ctor: b dims ok");
    CHECK(b.numel() == 6, "Move ctor: b numel ok");
    CHECK_EQ_DOUBLE(b.at(1,2), 5.0, "Move ctor: b data ok");

    // a queda "movido": estado válido. Según tu implementación, size_=0 y data_=nullptr
    CHECK(a.numel() == 0, "Move ctor: a.numel() debe ser 0 tras move");
}

void test_move_assignment() {
    std::vector<std::size_t> s = shape1(4);
    std::vector<double> v = linspace_values(4, 7.0);
    Tensor a(s, v);

    std::vector<std::size_t> s2 = shape2(2,2);
    std::vector<double> v2 = linspace_values(4, 0.0);
    Tensor b(s2, v2);

    b = std::move(a);

    CHECK(b.dims() == 1, "Move assign: b dims ok");
    CHECK(b.numel() == 4, "Move assign: b numel ok");
    CHECK_EQ_DOUBLE(b.at(3), 10.0, "Move assign: b data ok");
    CHECK(a.numel() == 0, "Move assign: a.numel() debe ser 0 tras move");
}

void test_exceptions() {
    // shape vacía
    try {
        std::vector<std::size_t> s;
        std::vector<double> v;
        Tensor t(s, v);
        CHECK(false, "Exception: shape vacía debió lanzar invalid_argument");
    } catch (const std::invalid_argument&) {
        CHECK(true, "Exception: shape vacía OK");
    }

    // dimensión 0
    try {
        std::vector<std::size_t> s = shape2(2, 0);
        std::vector<double> v;
        Tensor t(s, v);
        CHECK(false, "Exception: dimensión 0 debió lanzar invalid_argument");
    } catch (const std::invalid_argument&) {
        CHECK(true, "Exception: dimensión 0 OK");
    }

    // out of range
    try {
        std::vector<std::size_t> s = shape1(3);
        std::vector<double> v = linspace_values(3, 0.0);
        Tensor t(s, v);
        (void)t.at(3); // fuera de rango
        CHECK(false, "Exception: at(3) debió lanzar out_of_range");
    } catch (const std::out_of_range&) {
        CHECK(true, "Exception: out_of_range OK");
    }
}

// Si YA implementaste los creadores (zeros/ones/random/arange):
void test_creators_if_exist() {
    // zeros
    {
        std::vector<std::size_t> s = shape2(2,3);
        Tensor z = Tensor::zeros(s);
        CHECK(z.numel() == 6, "zeros: numel");
        CHECK_EQ_DOUBLE(z.at(1,2), 0.0, "zeros: valores 0");
    }

    // ones
    {
        std::vector<std::size_t> s = shape3(1,2,2);
        Tensor o = Tensor::ones(s);
        CHECK(o.numel() == 4, "ones: numel");
        CHECK_EQ_DOUBLE(o.at(0,1,1), 1.0, "ones: valores 1");
    }

    // arange
    {
        Tensor a = Tensor::arange(5, 10); // 5,6,7,8,9
        CHECK(a.dims() == 1, "arange: dims 1");
        CHECK(a.numel() == 5, "arange: numel 5");
        CHECK_EQ_DOUBLE(a.at(0), 5.0, "arange: at(0)=5");
        CHECK_EQ_DOUBLE(a.at(4), 9.0, "arange: at(4)=9");
    }

    // random (rango)
    {
        std::vector<std::size_t> s = shape1(20);
        // para test reproducible:
        std::srand(123);

        Tensor r = Tensor::random(s, -2.0, 3.0);
        for (std::size_t i = 0; i < r.numel(); ++i) {
            double x = r.at(i);
            CHECK(x >= -2.0 && x <= 3.0, "random: valor fuera de rango");
        }
    }
}

int main() {
    test_constructor_and_at_1d();
    test_constructor_and_at_2d_mapping();
    test_constructor_and_at_3d_mapping();

    test_copy_constructor_deep_copy();
    test_copy_assignment_deep_copy();
    test_move_constructor();
    test_move_assignment();

    test_exceptions();

    // Descomenta si ya tienes esos métodos static implementados
    test_creators_if_exist();

    std::cout << "\nPASSED: " << g_pass << "\nFAILED: " << g_fail << "\n";

    // Retorno típico: 0 si todo bien
    return (g_fail == 0) ? 0 : 1;
}
