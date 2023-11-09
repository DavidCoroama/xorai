#pragma once
#ifndef XORAI_NETWORK_H
#define XORAI_NETWORK_H

#include <xorai/matrix.h>
#include <xorai/model.h>

template<typename Float>
class Network
{
private:
    using matrix_t = Matrix<Float>;

public:
    explicit Network(const U64Array&, Float = 0.5);
    explicit Network(std::string, Float = 0.5);
    ~Network();

    matrix_t* feed_forward(matrix_t*);
    void back_propagate(matrix_t*, matrix_t*);
    void train(Dataset<Float>&, Dataset<Float>&, u64);
    void save(std::string, i8 = 8) const;
    matrix_t* test(Float, Float);

    U64Array layers;
    MatrixArray<Float> data;
    MatrixArray<Float> biases;
    MatrixArray<Float> weights;
    Float learning_rate;

private:
    void assert_float_type();
};

#endif //XORAI_NETWORK_H