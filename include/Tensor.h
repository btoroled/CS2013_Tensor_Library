//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#ifndef CS2013_TENSOR_LIBRARY_TENSOR_H
#define CS2013_TENSOR_LIBRARY_TENSOR_H
#include <vector>
#include "TensorTransform.h"



class Tensor {

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_ = 0;
    double* data_ = nullptr;

    static std::size_t product(const std::vector<std::size_t>& shape);
    void validate_shape_or_throw(const std::vector<std::size_t>& shape)const;
    void compute_strides();


    std::size_t offset(std::size_t i) const;
    std::size_t offset(std::size_t i, std::size_t j) const;
    std::size_t offset(std::size_t i, std::size_t j, std::size_t k) const;

    static std::vector<std::size_t> broadcast_shape_or_throw(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b );

public:
    //
    //CONSTRUCTORES
    //
    Tensor();
    Tensor(const std::vector<std::size_t>& shape_, const std::vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();


    const std::vector<std::size_t>& shape() const {return shape_;}
    std::size_t dims() const {return shape_.size();}
    std::size_t numel() const {return size_;}

    double& at(std::size_t i);
    double& at(std::size_t i, std::size_t j);
    double& at(std::size_t i, std::size_t j, std::size_t k);

    const double& at(std::size_t i) const;
    const double& at(std::size_t i, std::size_t j) const;
    const double& at(std::size_t i, std::size_t j, std::size_t k) const;

    //
    //METODOS
    //
    void imprimir() const;

    static Tensor zeros (const std::vector<std::size_t>& shape);
    static Tensor ones  (const std::vector<std::size_t>& shape);
    static Tensor random(const std::vector<std::size_t>& shape, double min, double max);
    static Tensor arange(long long start, long long end);

    Tensor view(const std::vector<std::size_t>& new_shape);
    Tensor unsqueeze(std::size_t dim);

    static Tensor concat(const std::vector<Tensor>& tensors, std::size_t dim);

    //
    //Friends
    //

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    //
    //Sobrecarga de operadores
    //

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    //Polimorfismo
    Tensor apply(const TensorTransform& op) const;

};





#endif //CS2013_TENSOR_LIBRARY_TENSOR_H