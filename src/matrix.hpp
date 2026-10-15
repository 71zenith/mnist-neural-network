#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

class Matrix {
public:
  size_t rows, cols;
  std::vector<double> data;

  Matrix() : rows(0), cols(0) {}
  Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

  double &operator()(size_t r, size_t c) { return data[r * cols + c]; }

  double operator()(size_t r, size_t c) const { return data[r * cols + c]; }

  void printShape() const { printf("(%zu, %zu)\n", rows, cols); }

  Matrix row(size_t i) const {
    assert(i < rows);
    Matrix r(1, cols);
    std::copy(data.begin() + i * cols, data.begin() + (i + 1) * cols,
              r.data.begin());
    return r;
  }

  Matrix random() const {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / rows));
    Matrix m = *this;
    for (auto &x : m.data)
      x = d(gen);
    return m;
  }
};

inline Matrix operator*(const Matrix &A, const double &b) {
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) * b;
    }
  }
  return C;
}

inline Matrix operator/(const Matrix &A, const double &b) {
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) / b;
    }
  }
  return C;
}

inline Matrix operator+(const Matrix &A, const double &b) {
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) + b;
    }
  }
  return C;
}

// matrix - scaler
inline Matrix operator-(const Matrix &A, const double &b) {
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) - b;
    }
  }
  return C;
}

// scaler - matrix
inline Matrix operator-(const double &b, const Matrix &A) {
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = b - A(i, j);
    }
  }
  return C;
}

// matrix cross matrix
inline Matrix operator*(const Matrix &A, const Matrix &B) {
  assert(A.cols == B.rows);
  Matrix C(A.rows, B.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t k = 0; k < A.cols; ++k) {
      for (size_t j = 0; j < B.cols; ++j) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return C;
}

// matrix + matrix
inline Matrix operator+(const Matrix &A, const Matrix &B) {
  assert(A.cols == B.cols && A.rows == B.rows);
  Matrix C(A.rows, B.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) + B(i, j);
    }
  }
  return C;
}

inline Matrix operator-(const Matrix &A, const Matrix &B) {
  assert(A.cols == B.cols && A.rows == B.rows);
  Matrix C(A.rows, B.cols);
  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      C(i, j) = A(i, j) - B(i, j);
    }
  }
  return C;
}

inline Matrix addRow(const Matrix &A, const Matrix &b) {
  assert(b.rows == 1 && b.cols == A.cols);
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.rows; ++i)
    for (size_t j = 0; j < A.cols; ++j)
      C(i, j) = A(i, j) + b(0, j);
  return C;
}

inline Matrix transpose(const Matrix &A) {
  Matrix T(A.cols, A.rows);
  for (size_t i = 0; i < A.rows; ++i)
    for (size_t j = 0; j < A.cols; ++j)
      T(j, i) = A(i, j);
  return T;
}

inline Matrix hadamard(const Matrix &A, const Matrix &B) {
  assert(A.rows == B.rows && A.cols == B.cols);
  Matrix C(A.rows, A.cols);
  for (size_t i = 0; i < A.data.size(); ++i)
    C.data[i] = A.data[i] * B.data[i];
  return C;
}
