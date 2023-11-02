#include <xorai/activation.h>
#include <cmath>

f64 sigmoid(f64 x)
{
    return 1.0 / (1.0 + expl(-x));
}

f64 derivative(f64 x)
{
    return x * (1.0 - x);
}