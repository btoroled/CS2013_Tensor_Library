//
// Created by Benjamin Toro Leddihn on 8/01/26.
//

#ifndef CS2013_TENSOR_LIBRARY_TENSOR_H
#define CS2013_TENSOR_LIBRARY_TENSOR_H
#include <vector>
#include <cstddef>
#include <cstddef>
#include <iostream>

class Tensor {

    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t size_ = 0;
    double* data = nullptr;

    static std::size_t product(const std::vector<std::size_t>& shape);
    void compute_strides();
    void validate_shape_or_throw(const std::vector<std::size_t>& shape)const;


    std::size_t offset(std::size_t i) const;
    std::size_t offset(std::size_t i, std::size_t j) const;
    std::size_t offset(std::size_t i, std::size_t j, std::size_t k) const;


    size_t total_size;
    double* data_=nullptr;

public:
    //
    //Constructores
    //
    Tensor(const std::vector<std::size_t>& shape_, const std::vector<double>& values);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    //
    //Destructor
    //
    ~Tensor();

    //
    //Sobrecargas
    //

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    //
    //Metodos
    //

    const std::vector<std::size_t>& shape() const {return shape_;}
    std::size_t dims() const {return shape_.size();}
    std::size_t numel() const {return size_;}

    double& at(std::size_t i);
    double& at(std::size_t i, std::size_t j);
    double& at(std::size_t i, std::size_t j, std::size_t k);

    const double& at(std::size_t i) const;
    const double& at(std::size_t i, std::size_t j) const;
    const double& at(std::size_t i, std::size_t j, std::size_t k) const;

    void imprimir() const;
};





#endif //CS2013_TENSOR_LIBRARY_TENSOR_H