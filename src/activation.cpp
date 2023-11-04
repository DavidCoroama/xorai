#include <xorai/activation.h>
#include <iostream>
#include <cmath>

#define INSTANTIATE_FUNCTION_FLOATS(f) \
    template f32 f<f32>(f32 x);        \
    template f64 f<f64>(f64 x);        \
    template f128 f<f128>(f128 x);

template<typename Float>
Float Exp(Float x)
{
    if constexpr (std::is_same_v<Float, f32>)
        return expf32(x);
    else if constexpr (std::is_same_v<Float, f64>)
        return expf64(x);
    else if constexpr (std::is_same_v<Float, f128>)
#ifdef __F128_SUPPORT__
        return expf128(x);
#else
        return expl(x);
#endif

    std::cout << "[C++ Activation]: Unsupported float type " << typeid(Float).name() << std::endl;
    exit(EXIT_FAILURE);
}

template<typename Float>
Float sigmoid(Float x)
{
    return 1.0 / (1.0 + Exp<Float>(-x));
}

template<typename Float>
Float derivative(Float x)
{
    return x * (1.0 - x);
}

INSTANTIATE_FUNCTION_FLOATS(Exp)
INSTANTIATE_FUNCTION_FLOATS(sigmoid)
INSTANTIATE_FUNCTION_FLOATS(derivative)