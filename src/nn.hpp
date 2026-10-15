#include "matrix.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Activation functions

inline Matrix Softmax(const Matrix &a) {
  Matrix r(a.rows, a.cols);
  for (std::size_t i = 0; i < a.rows; ++i) {
    double mx = a(i, 0);
    for (std::size_t j = 1; j < a.cols; ++j)
      mx = std::max(mx, a(i, j));
    double sum = 0.0;
    for (std::size_t j = 0; j < a.cols; ++j) {
      r(i, j) = std::exp(a(i, j) - mx);
      sum += r(i, j);
    }
    for (std::size_t j = 0; j < a.cols; ++j)
      r(i, j) /= sum;
  }
  return r;
}

inline Matrix ReLU(const Matrix &a) {
  Matrix result(a.rows, a.cols);
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    result.data[i] = std::max(0.0, a.data[i]);
  }
  return result;
}

inline Matrix DReLU(const Matrix &a) {
  Matrix result(a.rows, a.cols);
  for (std::size_t i = 0; i < a.data.size(); ++i) {
    result.data[i] = a.data[i] > 0 ? 1 : 0;
  }
  return result;
}
// Loss functions

inline double CategoricalCrossEntropy(const Matrix &pred,
                                      const Matrix &target) {
  assert(pred.rows == target.rows && pred.cols == target.cols);

  constexpr double eps = 1e-7;
  double loss = 0.0;

  for (std::size_t i = 0; i < pred.data.size(); ++i) {
    if (target.data[i] != 0.0) {
      loss -= target.data[i] * std::log(std::max(eps, pred.data[i]));
    }
  }
  return loss / pred.rows;
}

enum class Activation_Type { ReLU, Softmax };

inline Matrix activate(Activation_Type act, const Matrix &z) {
  switch (act) {
  case Activation_Type::ReLU:
    return ReLU(z);
  case Activation_Type::Softmax:
    return Softmax(z);
  }
  return z;
}

inline Matrix activateD(Activation_Type act, const Matrix &z) {
  switch (act) {
  case Activation_Type::ReLU:
    return DReLU(z);
  case Activation_Type::Softmax:
    return Softmax(z);
  }
  return z;
}

enum class Loss { CategoricalCrossEntropy };

inline double computeLoss(Loss loss, const Matrix &pred, const Matrix &target) {
  switch (loss) {
  case Loss::CategoricalCrossEntropy:
    return CategoricalCrossEntropy(pred, target);
  }
  return 0.0;
}

inline std::size_t onehotdecode(const Matrix &a) {
  std::size_t actual = 0;
  for (std::size_t j = 1; j < a.cols; ++j) {
    if (a(0, j) > a(0, actual))
      actual = j;
  }
  return actual;
}
