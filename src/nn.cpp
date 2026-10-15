#include "nn.hpp"
#include "matrix.hpp"
#include "parser.hpp"

struct Layer {
  Activation_Type activation;
  Matrix weights;
  Matrix biases;
};

class NN {
public:
  std::vector<Layer> layers;
  Loss loss_fn;
  double loss;
  double learning_rate = 0.05;
  size_t epoch;
  std::vector<Matrix> activations;

  NN(std::vector<Layer> layers, Loss loss_fn, std::size_t epoch)
      : epoch(epoch), loss_fn(loss_fn), layers(std::move(layers)) {}

  void fit(Matrix &X, Matrix &y) {
    for (size_t e = 0; e < epoch; ++e) {
      for (size_t i = 0; i < X.rows; ++i) {
        Matrix output = forward(X.row(i));
        double cost = calculate_loss(output, y.row(i));
        backward(y.row(i));
      }
    }
  }

  size_t predict(const Matrix &X) {
    Matrix output = forward(X);
    size_t best = 0;
    for (size_t j = 1; j < output.cols; ++j) {
      if (output(0, j) > output(0, best))
        best = j;
    }
    return best;
  }

  Matrix forward(const Matrix &X) {
    activations.clear();
    activations.push_back(X);
    Matrix output = X;
    for (const auto &layer : layers) {
      output = addRow((output * layer.weights), layer.biases);
      output = activate(layer.activation, output);
      activations.push_back(output);
    }
    return output;
  }

  double calculate_loss(const Matrix &output, const Matrix &target) {
    return computeLoss(loss_fn, output, target);
  }

  void backward(const Matrix &y) {
    Matrix delta = activations.back() - y;
    for (size_t i = layers.size(); i-- > 0;) {
      Matrix dW = transpose(activations[i]) * delta;
      Matrix db = delta;
      if (i > 0) {
        Matrix next = delta * transpose(layers[i].weights);
        delta =
            hadamard(next, activateD(layers[i - 1].activation, activations[i]));
      }
      layers[i].weights = layers[i].weights - dW * learning_rate;
      layers[i].biases = layers[i].biases - db * learning_rate;
    }
  }
};

int main() {
  auto training = parse_training_data();
  auto pixels = training.image_ndim_cols * training.image_ndim_rows;
  auto output = 10;
  auto hidden = 128;
  auto myNN = NN({{Activation_Type::ReLU,
                   {Matrix(pixels, hidden).random()},
                   {Matrix(1, hidden)}},
                  {Activation_Type::Softmax,
                   {Matrix(hidden, output).random()},
                   {Matrix(1, output)}}},
                 Loss::CategoricalCrossEntropy, 5);
  myNN.fit(training.images, training.target);

  auto testing = parse_test_data();
  size_t correct = 0;
  for (size_t i = 0; i < testing.count; ++i) {
    size_t guess = myNN.predict(testing.images.row(i));
    size_t actual = onehotdecode(testing.target.row(i));
    if (guess == actual)
      correct++;
  }

  printf("Accuracy: %f\n", (double)correct / testing.count);
};
