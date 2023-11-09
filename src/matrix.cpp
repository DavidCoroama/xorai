#include <xorai/matrix.h>
#include <iostream>
#include <cassert>
#include <random>

#ifdef __F128_SUPPORT__
extern "C" {
    #include <quadmath.h>
}
#endif

#define matrix_t Matrix<Float>

template<typename Float>
Matrix<Float>::Matrix(u64 rows, u64 cols, FloatArray& data)
    : rows(rows), cols(cols), data(data)
{
    assert_float_type();
    assert(data.size() - 1 != rows * cols);
}

template<typename Float>
matrix_t* Matrix<Float>::add(matrix_t* other)
{
    assert(this->rows == other->rows && this->cols == other->cols);

    FloatArray buffer = FloatArray::with_capacity(this->rows * this->cols)
            .fill([this, other]BASIC_UNARY(i, this->data[i] + other->data[i]));

    check_destroy(other);
    return make(buffer);
}

template<typename Float>
matrix_t* Matrix<Float>::sub(matrix_t* other)
{
    assert(this->rows == other->rows && cols == other->cols);

    FloatArray buffer = FloatArray::with_capacity(this->rows * this->cols)
            .fill([this, other]BASIC_UNARY(i, this->data[i] - other->data[i]));

    check_destroy(other);
    return make(buffer);
}

template<typename Float>
matrix_t* Matrix<Float>::mul(matrix_t* other)
{
    assert(this->rows == other->rows && this->cols == other->cols);

    FloatArray result(this->rows * this->cols, 0.0);

    for(u64 i = 0; i < this->data.size(); i++)
        result[i] = this->data[i] * other->data[i];

    check_destroy(other);
    return make(result);
}

template<typename Float>
matrix_t* Matrix<Float>::dot(matrix_t* other)
{
    assert(this->cols == other->rows);

    u64 i, j, k;
    Float sum;

    FloatArray result(this->rows * other->cols, 0.0);

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

template<typename Float>
matrix_t* Matrix<Float>::map(std::function<Float(Float)> func)
{
    return make(this->data.map(std::move(func)));
}

template<typename Float>
matrix_t* Matrix<Float>::ref()
{
    this->destroy = true;
    return this;
}

template<typename Float>
matrix_t* Matrix<Float>::clone(bool temporary) const
{
    return make(this->rows, this->cols, this->data, temporary);
}

template<typename Float>
matrix_t* Matrix<Float>::transpose()
{
    FloatArray buffer(this->rows * this->cols, 0.0);

    for(u64 i = 0; i < this->rows; i++)
        for(u64 j = 0; j < this->cols; j++)
            buffer[j * this->rows + i] = this->data[i * this->cols + j];

    return make(this->cols, this->rows, buffer);
}

template<typename Float>
matrix_t* Matrix<Float>::from(const FloatArray& data)
{
    return matrix_t::make(data.size(), 1, data, false);
}

template<typename Float>
matrix_t* Matrix<Float>::random(u64 rows, u64 cols)
{
#ifdef __F128_SUPPORT__
    if constexpr (std::is_same_v<Float, f128>)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<long double> dist(0.0, 1.0);

        FloatArray buffer = FloatArray::with_capacity(rows * cols)
                .fill([&dist, &gen]BASIC_UNARY(_, static_cast<Float>(dist(gen))));

        return matrix_t::make(rows, cols, buffer, false);
    }
    else
    {
#endif
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Float> dist(0.0, 1.0);

    FloatArray buffer = FloatArray::with_capacity(rows * cols)
            .fill([&dist, &gen]BASIC_UNARY(_, dist(gen)));

    return matrix_t::make(rows, cols, buffer, false);
#ifdef __F128_SUPPORT__
    }
#endif
}

template<typename Float>
void Matrix<Float>::display(matrix_t* matrix)
{
    for(u64 row = 0; row < matrix->rows; row++)
    {
        for(u64 col = 0; col < matrix->cols; col++)
        {
            std::cout << matrix_t::float_to_string(matrix->data[row * matrix->cols + col]);

            if(col < matrix->cols - 1)
                std::cout << "\t";
        }

        std::cout << "\n";
    }
}

template<typename Float>
void Matrix<Float>::check_destroy(matrix_t* matrix)
{
    if(matrix->destroy)
        delete(matrix);
}

template<typename Float>
std::string Matrix<Float>::float_to_string(Float number)
{
#ifdef __F128_SUPPORT__
    if constexpr (std::is_same_v<Float, f128>)
    {
        char buffer[128];
        int result = quadmath_snprintf(buffer, sizeof(buffer), "%.36Qg", number);

        return result >= 0 ? std::move(std::string(buffer)) : float_to_string(static_cast<f64>(number));
    }
    else
#endif
    return std::to_string(number);
}

template<typename Float>
matrix_t* Matrix<Float>::make(FloatArray _data)
{
    this->data = std::move(_data);
    return this;
}

template<typename Float>
matrix_t* Matrix<Float>::make(u64 _rows, u64 _cols, FloatArray& _data)
{
    this->rows = _rows;
    this->cols = _cols;
    this->data = _data;
    return this;
}

template<typename Float>
matrix_t* Matrix<Float>::make(u64 _rows, u64 _cols, FloatArray _data, bool _destroy)
{
    auto matrix = new matrix_t(_rows, _cols, _data);
    matrix->destroy = _destroy;
    return matrix;
}

template<typename Float>
void Matrix<Float>::assert_float_type()
{
    if constexpr (!is_float_type<Float>)
    {
        std::cout << "[C++ Matrix]: Matrix<Float> requires Float to be a floating point type.\n";
        std::cout << "\tSupported float types include: [f32 (float), f64 (double), and f128 (long double || __float128)]" << std::endl;
        exit(EXIT_FAILURE);
    }
}

INSTANTIATE_CLASS_FLOATS(Matrix)