#pragma once

#include <random>
#include <utility>
#include <numeric>
#include <limits>

namespace reference {

template <typename T>
struct sequence_initializer {
  static const bool kIsIntegral = std::is_integral_v<T>;
  using ElementType = T;
  using ElementInitType =
      typename std::conditional<kIsIntegral, int, float>::type;

  static void init(ElementType *data, int size,
                   ElementInitType start = ElementInitType(0),
                   ElementInitType step = ElementInitType(1)) {
    for (int i = 0; i < size; ++i) {
      data[i] = static_cast<ElementType>(start + i * step);
    }
  }
};

template <typename T>
struct random_initializer {
  static const bool kIsIntegral = std::is_integral_v<T>;
  using ElementType = T;
  using ElementInitType =
      typename std::conditional<kIsIntegral, int, float>::type;

  static void init(ElementType *data, int size,
                   ElementInitType low = ElementInitType(-1),
                   ElementInitType high = ElementInitType(1)) {
    std::random_device rd;
    std::default_random_engine eng(rd());

    /// Intergral.
    if constexpr (kIsIntegral) {
      std::uniform_int_distribution<ElementInitType> distr(low, high);

      for (int i = 0; i < size; ++i) {
        ElementInitType tmp = distr(eng);
        data[i] = tmp > std::numeric_limits<ElementType>::max() ?
            std::numeric_limits<ElementType>::max() :
            (tmp < std::numeric_limits<ElementType>::lowest() ? std::numeric_limits<ElementType>::lowest() : tmp);
      }

    /// Floating point.
    } else {
      std::uniform_real_distribution<ElementInitType> distr(low, high);

      for (int i = 0; i < size; ++i) {
        data[i] = static_cast<ElementType>(distr(eng));
      }
    }
  }
};

template <typename T>
struct diagonal_initializer {
  static const bool kIsIntegral = std::is_integral_v<T>;
  using ElementType = T;
  using ElementInitType =
      typename std::conditional<kIsIntegral, int, float>::type;

  static void init(ElementType *data, int rank,
                   ElementInitType value = ElementInitType(1)) {
    for (int i = 0; i < rank; ++i) {
      for (int j = 0; j < rank; ++j) {
        data[i * rank + j] = (i == j) ? ElementType(value) : ElementType(0);
      }
    }
  }
};

}