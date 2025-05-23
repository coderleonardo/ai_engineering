#include <torch/torch.h> // Includes the LibTorch library
#include <iostream>
#include <vector>
#include <cmath>

// --- Definition of NeuralNetwork Class using torch::nn::Module ---
class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_to_hidden(torch::nn::Linear(input_size, hidden_size)),
          hidden_to_output(torch::nn::Linear(hidden_size, output_size))
    {
        std::cout << "Initializing NeuralNetwork with LibTorch..." << std::endl;
        std::cout << "Input: " << input_size
                  << ", Hidden: " << hidden_size
                  << ", Output: " << output_size << std::endl;
        std::cout << "Neural Network built. Weights and biases initialized automatically by LibTorch." << std::endl;
    }

    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor hidden_output = torch::relu(input_to_hidden(x));
        torch::Tensor output = hidden_to_output(hidden_output);
        return output;
    }

private:
    torch::nn::Linear input_to_hidden;
    torch::nn::Linear hidden_to_output;
};

// --- Main Function (main) with Training Example ---
int main() {
    const int input_size = 1;
    const int hidden_size = 2;
    const int output_size = 1;
    const double learning_rate = 0.01;
    const int num_epochs = 1000;

    auto model = std::make_shared<NeuralNetwork>(input_size, hidden_size, output_size);

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Moving model to GPU." << std::endl;
        model->to(torch::kCUDA);
    } else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
    }

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    torch::Tensor x_train = torch::tensor({{1.0}, {2.0}, {3.0}, {4.0}});
    torch::Tensor y_train = torch::tensor({{3.0}, {5.0}, {7.0}, {9.0}});

    if (torch::cuda::is_available()) {
        x_train = x_train.to(torch::kCUDA);
        y_train = y_train.to(torch::kCUDA);
    }

    std::cout << "\nStarting Training..." << std::endl;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        torch::Tensor predictions = model->forward(x_train);
        torch::Tensor loss = torch::mse_loss(predictions, y_train);
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 100 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "], Loss: " << loss.item<double>() << std::endl;
        }
    }
    std::cout << "Training Completed." << std::endl;

    std::cout << "\nTesting the trained model..." << std::endl;
    torch::NoGradGuard no_grad;

    torch::Tensor test_input = torch::tensor({{5.0}});
    if (torch::cuda::is_available()) {
        test_input = test_input.to(torch::kCUDA);
    }
    torch::Tensor final_prediction = model->forward(test_input);
    std::cout << "Prediction for 5.0: " << final_prediction.item<double>() << " (Expected: 11.0)" << std::endl;

    torch::Tensor test_input_0 = torch::tensor({{0.0}});
    if (torch::cuda::is_available()) {
        test_input_0 = test_input_0.to(torch::kCUDA);
    }
    torch::Tensor final_prediction_0 = model->forward(test_input_0);
    std::cout << "Prediction for 0.0: " << final_prediction_0.item<double>() << " (Expected: 1.0)" << std::endl;

    return 0;
}