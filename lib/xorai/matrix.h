#pragma once
#ifndef XORAI_MATRIX_H
#define XORAI_MATRIX_H

#include <xorai/types.h>
#include <functional>
#include <iostream>

template<typename Float>
class Matrix
{
private:
    typedef cvector<Float> FloatArray;
    typedef Matrix<Float> matrix_t;

public:
    Matrix(u64, u64, FloatArray&);

    matrix_t* add(matrix_t*);
    matrix_t* sub(matrix_t*);
    matrix_t* mul(matrix_t*);
    matrix_t* dot(matrix_t*);
    matrix_t* map(std::function<Float(Float)>);
    matrix_t* ref();
    matrix_t* clone(bool = false) const;
    matrix_t* transpose();

    static matrix_t* from(const FloatArray&);
    static matrix_t* random(u64, u64);
    static void display(matrix_t*);

    u64 rows;
    u64 cols;
    FloatArray data;

private:
    static void check_destroy(matrix_t*);
    static std::string float_to_string(Float);

    matrix_t* make(FloatArray);
    matrix_t* make(u64, u64, FloatArray&);
    static matrix_t* make(u64, u64, FloatArray, bool);

    bool destroy = false;
};


template<typename Float>
using MatrixArray = cvector<Matrix<Float>*>;

#endif //XORAI_MATRIX_H