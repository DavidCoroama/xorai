#include <xorai/matrix.h>
#include <iostream>
#include <cassert>
#include <random>

Matrix::Matrix(u64 rows, u64 cols, F64Array& data)
    : rows(rows), cols(cols), data(data)
{
    assert(data.size() - 1 != rows * cols);
}

Matrix* Matrix::add(Matrix* other)
{
    assert(this->rows == other->rows && this->cols == other->cols);

    F64Array buffer = F64Array::with_capacity(this->rows * this->cols)
            .fill([this, other]BASIC_UNARY(i, this->data[i] + other->data[i]));

    check_destroy(other);
    return make(buffer);
}

Matrix* Matrix::sub(Matrix* other)
{
    assert(this->rows == other->rows && cols == other->cols);

    F64Array buffer = F64Array::with_capacity(this->rows * this->cols)
            .fill([this, other]BASIC_UNARY(i, this->data[i] - other->data[i]));

    check_destroy(other);
    return make(buffer);
}

Matrix* Matrix::mul(Matrix* other)
{
    assert(this->rows == other->rows && this->cols == other->cols);

    F64Array result(this->rows * this->cols, 0.0);

    for(u64 i = 0; i < this->data.size(); i++)
        result[i] = this->data[i] * other->data[i];

    check_destroy(other);
    return make(result);
}

Matrix* Matrix::dot(Matrix* other)
{
    assert(this->cols == other->rows);

    u64 i, j, k;
    f64 sum;

    F64Array result(this->rows * other->cols, 0.0);

    for(i = 0; i < this->rows; i++)
    {
        for(j = 0; j < other->cols; j++)
        {
            sum = 0.0;

            for(k = 0; k < this->cols; k++)
                sum += this->data[i * this->cols + k] * other->data[k * other->cols + j];

            result[i * other->cols + j] = sum;
        }
    }

    make(this->rows, other->cols, result);
    check_destroy(other);

    return this;
}

Matrix* Matrix::map(std::function<f64(f64)> func)
{
    return make(this->data.map(std::move(func)));
}

Matrix* Matrix::ref()
{
    this->destroy = true;
    return this;
}

Matrix* Matrix::clone(bool temporary) const
{
    return make(this->rows, this->cols, this->data, temporary);
}

Matrix* Matrix::transpose()
{
    F64Array buffer(this->rows * this->cols, 0.0);

    for(u64 i = 0; i < this->rows; i++)
        for(u64 j = 0; j < this->cols; j++)
            buffer[j * this->rows + i] = this->data[i * this->cols + j];

    return make(this->cols, this->rows, buffer);
}

Matrix* Matrix::from(const F64Array& data)
{
    return Matrix::make(data.size(), 1, data, false);
}

Matrix* Matrix::random(u64 rows, u64 cols)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<f64> dist(0.0, 1.0);

    auto buffer = F64Array::with_capacity(rows * cols)
            .fill([&dist, &gen]BASIC_UNARY(_, dist(gen)));

    return Matrix::make(rows, cols, buffer, false);
}

void Matrix::display(Matrix* matrix)
{
    for(u64 row = 0; row < matrix->rows; row++)
    {
        for(u64 col = 0; col < matrix->cols; col++)
        {
            std::cout << matrix->data[row * matrix->cols + col];

            if(col < matrix->cols - 1)
                std::cout << "\t";
        }

        std::cout << "\n";
    }
}

void Matrix::check_destroy(Matrix* matrix)
{
    if(matrix->destroy)
        delete(matrix);
}

Matrix* Matrix::make(F64Array _data)
{
    this->data = std::move(_data);
    return this;
}

Matrix* Matrix::make(u64 _rows, u64 _cols, F64Array& _data)
{
    this->rows = _rows;
    this->cols = _cols;
    this->data = _data;
    return this;
}

Matrix* Matrix::make(u64 _rows, u64 _cols, F64Array _data, bool _destroy)
{
    auto matrix = new Matrix(_rows, _cols, _data);
    matrix->destroy = _destroy;
    return matrix;
}