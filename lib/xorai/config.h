#pragma once
#ifndef XORAI_CONFIG_H
#define XORAI_CONFIG_H

/* Adds comments to `stdout` during network training. */
//#define DEBUG

/* Strictly prevents debug mode. */
//#define NO_DEBUG

/* Enables 128-bit floating point numbers.
 * If your system is not compatible with __float128,
 * it will be undefined automatically. */
//#define __F128_SUPPORT__

#endif //XORAI_CONFIG_H