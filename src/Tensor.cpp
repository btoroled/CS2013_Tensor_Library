//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#include "../include/Tensor.h"
#include <iostream>
#include <utility>
#include <cstdlib>
#include <stdexcept>


//
//Constructor VACIO
//
Tensor::Tensor() : shape_(), strides_(), size_(0), data_(nullptr) {}

//
//PRODuCTO DE VECTORES
//

std::size_t Tensor::product(const std::vector<std::size_t> &shape) {
    std::size_t p=1;
    for (std::size_t x :shape) p*=x;
    return p;
}

//
//VALIDACION DEl VECTOR
//
void Tensor::validate_shape_or_throw(const std::vector<std::size_t> &shape) const {
    if (shape.empty() || shape.size() > 3)
        throw std::invalid_argument("Tensor: shape tiene que tener de 1 a 3 dimensiones");
    for (std::size_t d :shape) {
        if (d==0) throw std::invalid_argument("Tensor: Las dimensiones de shape deben ser mayores a 0 < x y menores que x<3");
    }
}

void Tensor::compute_strides() {
    strides_.assign(shape_.size(), 1);

    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}


//
//CONSTRUCTOR PRINCIPAL
//

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::vector<double>& values)
    : shape_(shape) {

    validate_shape_or_throw(shape_);
    size_ = product(shape_);

    if (values.size() != size_) {
        throw std::invalid_argument("Tensor: values size does not match shape product");
    }

    compute_strides();

    data_ = new double[size_];
    for (std::size_t i = 0; i < size_; ++i)
        data_[i] = values[i];

}

//
// DESTRUCTOR
//

Tensor::~Tensor() {
    delete[] data_;
}

//
//CONSTRUCTOR DE COPIA
//

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(other.strides_),
      size_(other.size_),
      data_(nullptr) {

    if (size_ > 0) {
        data_ = new double[size_];
        for (std::size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }
}


Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    delete[] data_;

    shape_ = other.shape_;
    strides_ = other.strides_;
    size_ = other.size_;

    data_ = nullptr;
    if (size_ > 0) {
        data_ = new double[size_];
        for (std::size_t i = 0; i < size_; ++i) {
            data_[i] = other.data_[i];
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      size_(other.size_),
      data_(other.data_) {

    other.size_ = 0;
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    delete[] data_;

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    size_ = other.size_;
    data_ = other.data_;

    other.size_ = 0;
    other.data_ = nullptr;

    return *this;
}

std::size_t Tensor::offset(std::size_t i) const {
    if (shape_.size() != 1) {
        throw std::invalid_argument("Tensor: expected 1D");
    }
    if (i >= shape_[0]) {
        throw std::out_of_range("Tensor: index out of range");
    }
    return i;
}

std::size_t Tensor::offset(std::size_t i, std::size_t j) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Tensor: expected 2D");
    }
    if (i >= shape_[0] || j >= shape_[1]) {
        throw std::out_of_range("Tensor: index out of range");
    }
    return i * strides_[0] + j * strides_[1];
}

std::size_t Tensor::offset(std::size_t i, std::size_t j, std::size_t k) const {
    if (shape_.size() != 3) {
        throw std::invalid_argument("Tensor: expected 3D");
    }
    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
        throw std::out_of_range("Tensor: index out of range");
    }
    return i * strides_[0] + j * strides_[1] + k * strides_[2];
}

double& Tensor::at(std::size_t i) {
    return data_[offset(i)];
}

double& Tensor::at(std::size_t i, std::size_t j) {
    return data_[offset(i, j)];
}

double& Tensor::at(std::size_t i, std::size_t j, std::size_t k) {
    return data_[offset(i, j, k)];
}

const double& Tensor::at(std::size_t i) const {
    return data_[offset(i)];
}

const double& Tensor::at(std::size_t i, std::size_t j) const {
    return data_[offset(i, j)];
}

const double& Tensor::at(std::size_t i, std::size_t j, std::size_t k) const {
    return data_[offset(i, j, k)];
}


//
//MATRICES AUTOMATICAS
//

Tensor Tensor::zeros(const std::vector<std::size_t>& shape) {

    Tensor t;
    t.validate_shape_or_throw(shape);
    t.shape_ = shape;
    t.size_ = product(shape);
    t.compute_strides();

    t.data_ = new double[t.size_];
    for (std::size_t i = 0; i < t.size_; ++i) t.data_[i] = 0.0;

    return t;
}

Tensor Tensor::ones(const std::vector<std::size_t> &shape) {

    Tensor t;
    t.validate_shape_or_throw(shape);
    t.shape_ = shape;
    t.size_ = product(shape);
    t.compute_strides();

    t.data_ = new double [t.size_];
    for (std::size_t i= 0; i < t.size_; i++) t.data_[i] = 1;

    return t;
}

Tensor Tensor::random(const std::vector<std::size_t> &shape, double min, double max) {
    if (!(min<max))
        throw std::invalid_argument("Tensor::random: min debe ser < max");
    Tensor t;
    t.validate_shape_or_throw(shape);
    t.shape_ = shape;
    t.size_ = product(shape);
    t.compute_strides();

    t.data_ = new double[t.size_];
    for (std::size_t i= 0; i < t.size_;i++) {
        double u = (double)rand() / (double)RAND_MAX;
        t.data_[i] = min + (max -min) * u;
    }

    return t;
}

Tensor Tensor::arange(long long start, long long end) {
    if (end <= start)
        throw std::invalid_argument("Tensor::arange: end debe ser > start.");

    std::size_t n =(std::size_t)(end-start);
    Tensor t;
    t.shape_.clear();
    t.shape_.push_back(n);
    t.validate_shape_or_throw(t.shape_);
    t.size_ = n;
    t.compute_strides();

    t.data_ = new double[n];
    for (std::size_t i = 0; i < n; ++i) {
        t.data_[i] = (double)(start + (long long)i);
    }
    return t;
}

//
//IMPLEMENTACION DE SOBRE CARGA
//

std::vector<std::size_t> Tensor::broadcast_shape_or_throw(const std::vector<std::size_t> &a, const std::vector<std::size_t> &b) {

    if (a.size() != b.size())
        throw std::invalid_argument("Tensor: dimensiones incompatibles(diferente cantidad de dimensiones)");

    std::vector<std::size_t> out;
    out.reserve(a.size());

    for (std::size_t d =0; d < a.size(); d++) {
        const std::size_t ad = a[d];
        const std::size_t bd = b[d];

        if (ad==bd)
            out.push_back(ad);
        else if (ad==1)
            out.push_back(bd);
        else if (bd==1)
            out.push_back(ad);
        else
            throw std::invalid_argument("Tensor: shape incompatible para broadcast");
    }
    return out;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor r;
    r.shape_ = shape_;
    r.strides_ = strides_;
    r.size_ = size_;

    if (r.size_>0) {
        r.data_ = new double[r.size_];
        for (std::size_t i=0; i < r.size_; i++)
            r.data_[i]=data_[i] *scalar;
    }
    return r;
}

Tensor Tensor::operator+(const Tensor &other) const {
    std::vector<std::size_t> out_shape = broadcast_shape_or_throw(shape_, other.shape_);
    Tensor r;
    r.validate_shape_or_throw(out_shape);
    r.shape_ = out_shape;
    r.size_ = product(out_shape);
    r.compute_strides();
    r.data_ = new double [r.size_];

    if (dims()==1) {
        for (std::size_t i= 0; i < out_shape[0]; i++) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0]== 1 ? 0 : i);
            r.data_[i]= data_[ia] + other.data_[ib];
        }
        return r;
    }

    if (dims()==2) {
        const std::size_t R = out_shape[0], C = out_shape[1];
        for (std::size_t i = 0; i < R; i++) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0]== 1 ? 0 : i);
            for (std::size_t j=0; j < C; j++) {
                std::size_t ja = (shape_[1] == 1 ? 0 : j);
                std::size_t jb = (other.shape_[1] == 1 ? 0 : j);

                std::size_t oa  = ia *strides_[0] + ja * strides_[1];
                std::size_t ob = ib * other.strides_[0]+jb *other.strides_[1];
                std::size_t orr = i * r.strides_[0] +j *r.strides_[1];

                r.data_[orr] = data_[oa] + other.data_[ob];
            }
        }
        return r;
    }

    const std::size_t A = out_shape[0], B = out_shape[1], C = out_shape[2];
    for (std::size_t i = 0; i < A; ++i) {
        std::size_t ia = (shape_[0] == 1 ? 0 : i);
        std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
        for (std::size_t j = 0; j < B; ++j) {
            std::size_t ja = (shape_[1] == 1 ? 0 : j);
            std::size_t jb = (other.shape_[1] == 1 ? 0 : j);
            for (std::size_t k = 0; k < C; ++k) {
                std::size_t ka = (shape_[2] == 1 ? 0 : k);
                std::size_t kb = (other.shape_[2] == 1 ? 0 : k);

                std::size_t oa = ia * strides_[0] + ja * strides_[1] + ka * strides_[2];
                std::size_t ob = ib * other.strides_[0] + jb * other.strides_[1] + kb * other.strides_[2];
                std::size_t orr = i * r.strides_[0] + j * r.strides_[1] + k * r.strides_[2];

                r.data_[orr] = data_[oa] + other.data_[ob];
            }
        }
    }
    return r;
}

Tensor Tensor::operator-(const Tensor &other) const {
    std::vector<std::size_t> out_shape = broadcast_shape_or_throw(shape_, other.shape_);

    Tensor r;
    r.validate_shape_or_throw(out_shape);
    r.shape_ = out_shape;
    r.size_ = product(out_shape);
    r.compute_strides();
    r.data_ = new double[r.size_];

    if (dims() == 1) {
        for (std::size_t i = 0; i < out_shape[0]; ++i) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
            r.data_[i] = data_[ia] - other.data_[ib];
        }
        return r;
    }

    if (dims() == 2) {
        const std::size_t R = out_shape[0], C = out_shape[1];
        for (std::size_t i = 0; i < R; ++i) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
            for (std::size_t j = 0; j < C; ++j) {
                std::size_t ja = (shape_[1] == 1 ? 0 : j);
                std::size_t jb = (other.shape_[1] == 1 ? 0 : j);

                std::size_t oa = ia * strides_[0] + ja * strides_[1];
                std::size_t ob = ib * other.strides_[0] + jb * other.strides_[1];
                std::size_t orr = i * r.strides_[0] + j * r.strides_[1];

                r.data_[orr] = data_[oa] - other.data_[ob];
            }
        }
        return r;
    }

    const std::size_t A = out_shape[0], B = out_shape[1], C = out_shape[2];
    for (std::size_t i = 0; i < A; ++i) {
        std::size_t ia = (shape_[0] == 1 ? 0 : i);
        std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
        for (std::size_t j = 0; j < B; ++j) {
            std::size_t ja = (shape_[1] == 1 ? 0 : j);
            std::size_t jb = (other.shape_[1] == 1 ? 0 : j);
            for (std::size_t k = 0; k < C; ++k) {
                std::size_t ka = (shape_[2] == 1 ? 0 : k);
                std::size_t kb = (other.shape_[2] == 1 ? 0 : k);

                std::size_t oa = ia * strides_[0] + ja * strides_[1] + ka * strides_[2];
                std::size_t ob = ib * other.strides_[0] + jb * other.strides_[1] + kb * other.strides_[2];
                std::size_t orr = i * r.strides_[0] + j * r.strides_[1] + k * r.strides_[2];

                r.data_[orr] = data_[oa] - other.data_[ob];
            }
        }
    }
    return r;
}

Tensor Tensor::operator*(const Tensor& other) const {
    std::vector<std::size_t> out_shape = broadcast_shape_or_throw(shape_, other.shape_);

    Tensor r;
    r.validate_shape_or_throw(out_shape);
    r.shape_ = out_shape;
    r.size_ = product(out_shape);
    r.compute_strides();
    r.data_ = new double[r.size_];

    if (dims() == 1) {
        for (std::size_t i = 0; i < out_shape[0]; ++i) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
            r.data_[i] = data_[ia] * other.data_[ib];
        }
        return r;
    }

    if (dims() == 2) {
        const std::size_t R = out_shape[0], C = out_shape[1];
        for (std::size_t i = 0; i < R; ++i) {
            std::size_t ia = (shape_[0] == 1 ? 0 : i);
            std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
            for (std::size_t j = 0; j < C; ++j) {
                std::size_t ja = (shape_[1] == 1 ? 0 : j);
                std::size_t jb = (other.shape_[1] == 1 ? 0 : j);

                std::size_t oa = ia * strides_[0] + ja * strides_[1];
                std::size_t ob = ib * other.strides_[0] + jb * other.strides_[1];
                std::size_t orr = i * r.strides_[0] + j * r.strides_[1];

                r.data_[orr] = data_[oa] * other.data_[ob];
            }
        }
        return r;
    }

    const std::size_t A = out_shape[0], B = out_shape[1], C = out_shape[2];
    for (std::size_t i = 0; i < A; ++i) {
        std::size_t ia = (shape_[0] == 1 ? 0 : i);
        std::size_t ib = (other.shape_[0] == 1 ? 0 : i);
        for (std::size_t j = 0; j < B; ++j) {
            std::size_t ja = (shape_[1] == 1 ? 0 : j);
            std::size_t jb = (other.shape_[1] == 1 ? 0 : j);
            for (std::size_t k = 0; k < C; ++k) {
                std::size_t ka = (shape_[2] == 1 ? 0 : k);
                std::size_t kb = (other.shape_[2] == 1 ? 0 : k);

                std::size_t oa = ia * strides_[0] + ja * strides_[1] + ka * strides_[2];
                std::size_t ob = ib * other.strides_[0] + jb * other.strides_[1] + kb * other.strides_[2];
                std::size_t orr = i * r.strides_[0] + j * r.strides_[1] + k * r.strides_[2];

                r.data_[orr] = data_[oa] * other.data_[ob];
            }
        }
    }
    return r;
}


//
//Impresion de TENSORES
//

void Tensor::imprimir() const {
    if (dims() == 1) {
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            std::cout << at(i) << " ";
        }
        std::cout << "\n";
    } else if (dims() == 2) {
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            for (std::size_t j = 0; j < shape_[1]; ++j) {
                std::cout << at(i, j) << " ";
            }
            std::cout << "\n";
        }
    } else if (dims() == 3) {
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            std::cout << "Slice i=" << i << ":\n";
            for (std::size_t j = 0; j < shape_[1]; ++j) {
                for (std::size_t k = 0; k < shape_[2]; ++k) {
                    std::cout << at(i, j, k) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }



}
