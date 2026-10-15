#include "matrix.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

const std::string TRAIN_IMAGE_PATH = "../data/train-images.idx3-ubyte";
const std::string TRAIN_LABEL_PATH = "../data/train-labels.idx1-ubyte";
const std::string TEST_IMAGE_PATH = "../data/t10k-images.idx3-ubyte";
const std::string TEST_LABEL_PATH = "../data/t10k-labels.idx1-ubyte";

const uint32_t MAGIC_IMAGE = 0x00000803;
const uint32_t MAGIC_LABEL = 0x00000801;

struct MNIST {
  uint32_t count = 0;           // 60000
  uint32_t image_ndim_rows = 0; // 28
  uint32_t image_ndim_cols = 0; // 28
  Matrix images;                // count x 784, values from [0, 1]
  Matrix target;                // count x 10, one hot encoded
};

inline uint32_t read_32b(std::ifstream &f) {
  unsigned char b[4];
  f.read(reinterpret_cast<char *>(b), 4);
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

inline std::ifstream read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open " + path);
  return f;
}

inline MNIST parse_mnist(const std::string &imagePath,
                         const std::string &labelPath) {
  auto img = read_file(imagePath);
  auto lbl = read_file(labelPath);

  if (read_32b(img) != MAGIC_IMAGE)
    throw std::runtime_error("Bad image magic number");
  if (read_32b(lbl) != MAGIC_LABEL)
    throw std::runtime_error("Bad label magic number");

  MNIST m;
  m.count = read_32b(img);
  m.image_ndim_rows = read_32b(img);
  m.image_ndim_cols = read_32b(img);

  uint32_t labelCount = read_32b(lbl);
  if (labelCount != m.count)
    throw std::runtime_error("Image/label count mismatch");

  const std::size_t pixels = std::size_t(m.image_ndim_rows) * m.image_ndim_cols;
  std::vector<uint8_t> raw(m.count * pixels);
  img.read(reinterpret_cast<char *>(raw.data()), raw.size());
  if (!img)
    throw std::runtime_error("Truncated image file");

  m.images = Matrix(m.count, pixels);
  for (std::size_t i = 0; i < raw.size(); ++i)
    m.images.data[i] = raw[i] / 255.0;

  std::vector<uint8_t> labels(m.count);
  lbl.read(reinterpret_cast<char *>(labels.data()), labels.size());
  if (!lbl)
    throw std::runtime_error("Truncated label file");

  m.target = Matrix(m.count, 10);
  for (std::size_t i = 0; i < m.count; ++i)
    m.target.data[i * 10 + labels[i]] = 1.0;

  return m;
}

inline MNIST parse_training_data() {
  return parse_mnist(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH);
}

inline MNIST parse_test_data() {
  return parse_mnist(TEST_IMAGE_PATH, TEST_LABEL_PATH);
}
