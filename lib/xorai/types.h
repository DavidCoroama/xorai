#pragma once
#ifndef BENN_TYPES_H
#define BENN_TYPES_H

#include <vector>

#define BASIC_UNARY(arg, code) (const auto& arg) { return code; }
#define BASIC_UNARY_DELETE [](auto* a) { delete(a); }

#define S signed
#define U unsigned
#define T(v, n) typedef v n

T(U short,     u16);
T(U long,      u32);
T(U long long, u64);

T(S short,     i16);
T(S long,      i32);
T(S long long, i64);

T(float,       f16);
T(double,      f32);
T(long double, f64);

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

T(cvector<u64>,      U64Array);
T(cvector<f64>,      F64Array);
T(cvector<F64Array>, Dataset);

#endif //BENN_TYPES_H