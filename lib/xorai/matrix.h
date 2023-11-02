#pragma once
#ifndef BENN_MATRIX_H
#define BENN_MATRIX_H

#include <xorai/types.h>
#include <functional>
#include <iostream>

class Matrix
{
public:
    Matrix(u64, u64, F64Array&);

    Matrix* add(Matrix*);
    Matrix* sub(Matrix*);
    Matrix* mul(Matrix*);
    Matrix* dot(Matrix*);
    Matrix* map(std::function<f64(f64)>);
    Matrix* ref();
    Matrix* clone(bool = false) const;
    Matrix* transpose();

    static Matrix* from(const F64Array&);
    static Matrix* random(u64, u64);
    static void display(Matrix*);

    u64 rows;
    u64 cols;
    F64Array data;

private:
    static void check_destroy(Matrix*);
    Matrix* make(F64Array);
    Matrix* make(u64, u64, F64Array&);
    static Matrix* make(u64, u64, F64Array, bool);

    bool destroy = false;
};

T(cvector<Matrix*>, MatrixArray);

#undef S
#undef U
#undef T

#endif //BENN_MATRIX_H