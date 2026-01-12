//
// Created by Benjamin Toro Leddihn on 12/01/26.
//

#ifndef CS2013_TENSOR_LIBRARY_TENSORTRANSFORM_H
#define CS2013_TENSOR_LIBRARY_TENSORTRANSFORM_H
#include <cmath>

class TensorTransform {
public:
    virtual double apply(double x) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    double apply(double x) const override {
        return (x > 0.0) ? x : 0.0;
    }
};

class Sigmoid : public TensorTransform {
public:
    double apply(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
};



#endif //CS2013_TENSOR_LIBRARY_TENSORTRANSFORM_H