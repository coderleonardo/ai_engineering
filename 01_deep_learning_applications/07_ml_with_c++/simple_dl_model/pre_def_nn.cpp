#include <iostream>
#include <vector>   // Para usar std::vector
#include <cmath>    // Para usar std::max (para ReLU) e std::exp (se fosse sigmoid/softmax)

// --- Definição da Classe NeuralNetwork ---
class NeuralNetwork {
public:
    // Construtor: Inicializa a rede com o número de neurônios em cada camada
    // Para este exemplo, os pesos e biases são fixos, mas em um modelo real
    // seriam carregados de um arquivo ou inicializados aleatoriamente.
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_layer_size(input_size),
          hidden_layer_size(hidden_size),
          output_layer_size(output_size)
    {
        std::cout << "Inicializando NeuralNetwork..." << std::endl;
        std::cout << "Entrada: " << input_layer_size
                  << ", Oculta: " << hidden_layer_size
                  << ", Saída: " << output_layer_size << std::endl;

        // Para este exemplo, vamos definir alguns pesos e biases manuais.
        // Em um modelo real, estes seriam matrizes carregadas de um arquivo,
        // por exemplo, usando uma biblioteca de álgebra linear como Eigen.

        // Pesos da camada de entrada para a camada oculta (W_ih)
        // Dimensões: hidden_size x input_size
        // Exemplo: 2 neurônios ocultos, 1 entrada
        W_ih = {{0.5}, {0.8}}; // w1, w2 para a primeira e segunda neuronio oculto respectivamente

        // Biases da camada oculta (b_h)
        // Dimensões: hidden_size x 1
        b_h = {0.1, 0.2}; // b1, b2

        // Pesos da camada oculta para a camada de saída (W_ho)
        // Dimensões: output_size x hidden_size
        // Exemplo: 1 saída, 2 neurônios ocultos
        W_ho = {{0.3, 0.7}}; // w1_prime, w2_prime

        // Biases da camada de saída (b_o)
        // Dimensões: output_size x 1
        b_o = {0.05}; // b_output

        std::cout << "Rede Neural inicializada com pesos e biases pré-definidos." << std::endl;
    }

    // Destrutor: Não precisamos liberar memória alocada dinamicamente com std::vector neste caso,
    // mas é bom ter para demonstração e para classes mais complexas.
    ~NeuralNetwork() {
        std::cout << "Destrutor da NeuralNetwork chamado." << std::endl;
    }

    // Método para realizar a inferência (passagem forward)
    // Recebe um vetor de entrada (features) e retorna um vetor de saídas (previsões).
    // Usamos 'const std::vector<double>& input' para evitar cópias desnecessárias
    // e garantir que a entrada não seja modificada.
    std::vector<double> predict(const std::vector<double>& input) const {
        if (input.size() != input_layer_size) {
            std::cerr << "Erro: Tamanho da entrada (" << input.size()
                      << ") não corresponde ao esperado (" << input_layer_size << ")." << std::endl;
            return {}; // Retorna um vetor vazio em caso de erro
        }

        // --- Camada Oculta ---
        std::vector<double> hidden_output(hidden_layer_size);
        for (int i = 0; i < hidden_layer_size; ++i) {
            double activation_sum = 0.0;
            for (int j = 0; j < input_layer_size; ++j) {
                activation_sum += input[j] * W_ih[i][j];
            }
            activation_sum += b_h[i]; // Adiciona o bias

            // Função de ativação ReLU: max(0, x)
            hidden_output[i] = std::max(0.0, activation_sum);
        }

        // --- Camada de Saída (Linear) ---
        std::vector<double> output(output_layer_size);
        for (int i = 0; i < output_layer_size; ++i) {
            double final_sum = 0.0;
            for (int j = 0; j < hidden_layer_size; ++j) {
                final_sum += hidden_output[j] * W_ho[i][j];
            }
            final_sum += b_o[i]; // Adiciona o bias da camada de saída

            // Função de ativação linear (apenas o resultado da soma)
            output[i] = final_sum;
        }

        return output;
    }

private: // Membros privados: representam a arquitetura e os parâmetros da rede
    int input_layer_size;
    int hidden_layer_size;
    int output_layer_size;

    // Pesos e Biases (para simplicidade, representados como vectors de vectors)
    // Em produção, bibliotecas de álgebra linear (Eigen, Blaze) seriam usadas.
    std::vector<std::vector<double>> W_ih; // Pesos Input -> Hidden
    std::vector<double> b_h;              // Biases Hidden
    std::vector<std::vector<double>> W_ho; // Pesos Hidden -> Output
    std::vector<double> b_o;              // Biases Output
};

// --- Função Principal (main) para Testar a Classe ---
int main() {
    // Criando uma instância da nossa rede neural
    // 1 neurônio de entrada, 2 neurônios na camada oculta, 1 neurônio de saída
    NeuralNetwork my_nn(1, 2, 1);

    // Exemplo de entrada para previsão
    std::vector<double> input_data = {5.0}; // Nosso valor de entrada

    // Realizando a previsão
    std::vector<double> prediction = my_nn.predict(input_data);

    // Exibindo o resultado
    if (!prediction.empty()) {
        std::cout << "\nPrevisão para entrada [" << input_data[0] << "]: ";
        for (double val : prediction) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Exemplo com outra entrada
    std::vector<double> input_data_2 = {1.0};
    std::vector<double> prediction_2 = my_nn.predict(input_data_2);
    if (!prediction_2.empty()) {
        std::cout << "Previsão para entrada [" << input_data_2[0] << "]: ";
        for (double val : prediction_2) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }


    // O destrutor de 'my_nn' será chamado automaticamente ao final de main().
    return 0;
}