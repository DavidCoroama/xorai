#pragma once
#ifndef XORAI_TYPES_H
#define XORAI_TYPES_H

#include <xorai/config.h>
#include <vector>

#define BASIC_UNARY(arg, code) (const auto& arg) { return code; }
#define BASIC_UNARY_DELETE [](auto* a) { delete(a); }

#define INSTANTIATE_CLASS_FLOATS(c) \
    template class c<f32>;          \
    template class c<f64>;          \
    template class c<f128>;

#define S signed
#define U unsigned
#define T(v, n) typedef v n

T(S char,      i8);
T(S short int, i16);
T(S int,       i32);

T(U char,      u8);
T(U short int, u16);
T(U int,       u32);

#define MIN_F32_PRECISION 6
#define MAX_F32_PRECISION 9

#define MIN_F64_PRECISION 15
#define MAX_F64_PRECISION 17

T(float,       f32);
T(double,      f64);

#if defined(__SIZEOF_FLOAT128__) && defined(__F128_SUPPORT__)
T(__float128,  f128);

#define MIN_F128_PRECISION 18
#define MAX_F128_PRECISION 35
#else
T(long double, f128);

#define MIN_F128_PRECISION MIN_F64_PRECISION
#define MAX_F128_PRECISION MAX_F64_PRECISION
#endif

#define UseMinPrecision(bitsize) MIN_F ## bitsize ## _PRECISION
#define UseMaxPrecision(bitsize) MAX_F ## bitsize ## _PRECISION
#define UsePrecision(bitsize) UseMinPrecision(bitsize)

#if defined(__x86_64__) && !defined(__ILP32__)
T(U long int, u64);
T(S long int, i64);
#else
T(U long long int, u64);
T(S long long int, i64);
#endif

#ifdef __SIZEOF_INT128__
T(__int128_t,  i128);
T(__uint128_t, u128);
#endif

#if defined(__F128_SUPPORT__) && (!defined(__SIZEOF_FLOAT128__) && !defined(__STDCPP_FLOAT128_T__))
#undef __F128_SUPPORT__
#endif

template<typename _Tp, typename... _Types>
constexpr bool is_type_of = (std::is_same_v<_Tp, _Types> || ...);

template<typename _Tp>
constexpr bool is_float_type = is_type_of<_Tp, f32, f64, f128>;

template<typename _Tp, typename _Alloc = std::allocator<_Tp>>
class cvector : public std::vector<_Tp, _Alloc>
{
private:
    typedef cvector<_Tp, _Alloc> _SelfType;

    template<typename _UnaryOperation>
    class __cvector_map_details
    {
    public:
        using UnaryOpReturnType = typename std::result_of_t<_UnaryOperation(_Tp)>;
        using IsVoidReturnType = std::is_void<UnaryOpReturnType>;
        using IsSameReturnType = std::is_same<UnaryOpReturnType, _Tp>;
        using NewCvectorType = cvector<UnaryOpReturnType, std::allocator<UnaryOpReturnType>>;
        using NonVoidReturnType = std::conditional_t<IsSameReturnType::value, _SelfType, NewCvectorType>;

        static auto map(_SelfType* self, _UnaryOperation func, NonVoidReturnType vector)
        {
            for(auto it = vector.begin(); it != vector.end(); it++)
                *it = func(self->operator[](it - vector.begin()));

            return std::move(vector);
        };
    };

    size_t current_capacity = this->size();

public:
    using std::vector<_Tp, _Alloc>::vector;

    template<typename _UnaryOperation>
    _SelfType
    fill(_UnaryOperation func)
    {
        for(u64 i = 0; i < this->reserve_size(); i++)
            this->push_back(func(i));

        return *this;
    }

    _SelfType
    clone() const
    {
        _SelfType vector(*this);
        return vector;
    }

    size_t
    reserve_size()
    {
        return this->current_capacity;
    }

    static _SelfType
    with_capacity(size_t size)
    {
        _SelfType vector;
        vector.reserve(size);
        vector.current_capacity = size;
        return vector;
    }

    template<typename _UnaryOperation>
    std::conditional_t<
        !__cvector_map_details<_UnaryOperation>::IsVoidReturnType::value,
        typename __cvector_map_details<_UnaryOperation>::NonVoidReturnType,
        void
    >
    map(_UnaryOperation func)
    {
        using _Helper = __cvector_map_details<_UnaryOperation>;

        if constexpr (_Helper::IsSameReturnType::value)
        {
            for(auto it = this->begin(); it != this->end(); it++)
                *it = func(*it);

            return *this;
        }
        else if constexpr (_Helper::IsVoidReturnType::value)
        {
            for(auto it = this->begin(); it != this->end(); it++)
                func(*it);

            return;
        }
        else
        {
            if constexpr (_Helper::IsSameReturnType::value)
                return _Helper::map(this, func, _SelfType::with_capacity(this->size()));

            return _Helper::map(this, func, _Helper::NewCvectorType::with_capacity(this->size()));
        }
    }

    template<typename _UnaryOperation>
    auto map_if(bool condition, _UnaryOperation func)
    {
        if(condition)
            return map(func);
    }
};

T(cvector<u64>,  U64Array);
T(cvector<f32>,  F32Array);
T(cvector<f64>,  F64Array);
T(cvector<f128>, F128Array);

template<typename Float = f32>
using Dataset = cvector<cvector<Float>>;

#undef S
#undef U
#undef T

#endif //XORAI_TYPES_H