#pragma once
#ifndef XORAI_ACTIVATION_H
#define XORAI_ACTIVATION_H

#include <xorai/types.h>

template<typename Float>
Float Exp(Float);

template<typename Float>
Float sigmoid(Float);

template<typename Float>
Float derivative(Float);

#endif //XORAI_ACTIVATION_H