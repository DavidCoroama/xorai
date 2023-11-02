#pragma once
#ifndef BENN_NETWORK_H
#define BENN_NETWORK_H

#include <xorai/model.h>

/* Adds comments to `stdout` during network training. */
//#define DEBUG

/* Strictly prevents debug mode. */
//#define NO_DEBUG

class Network
{
public:
    Network(const U64Array&, f64 = 0.5);
    Network(std::string, f64 = 0.5);
    ~Network();

    Matrix* feed_forward(Matrix*);
    void back_propagate(Matrix*, Matrix*);
    void train(Dataset&, Dataset&, u64);
    void save(std::string, i64 = 7) const;
    Matrix* test(f64, f64);

    U64Array layers;
    MatrixArray data;
    MatrixArray biases;
    MatrixArray weights;
    f64 learning_rate;
};

#endif //BENN_NETWORK_H