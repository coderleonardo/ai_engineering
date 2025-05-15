// Document classification with Machine Learning

// warning suppression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

// libs
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <random>

using namespace std;
using namespace std::chrono;

const double TEST_SIZE_PCT = 0.3;
const int NUM_ITERATIONS = 5;

double calculate_mean(vector<double> vect);
double calculate_variance(vector<double> vect);

void print2d_vector(vector<vector<double>> vect);
vector<vector<double>> prior_probability(vector<double> vect);
vector<vector<double>> count_classes(vector<double>);
vector<vector<double>> document_likelihood(vector<double> class_vector, vector<double> doc_type, vector<vector<double>> class_count);
vector<vector<double>> certificated_likelihood(vector<double> class_vector, vector<double> valid_certificated, vector<vector<double>> class_count);
vector<vector<double>> days_used_mean(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count);
vector<vector<double>> days_used_variance(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count);
vector<vector<double>> days_used_metrics(vector<vector<double>> days_used_mean, vector<vector<double>> days_used_variance);

double calculate_days_used(double v, double mean_v, double variance_v);

// Bayes Theorem Method
vector<vector<double>> bayes_theorem(
    double doc_type, 
    double valid_certificated, 
    double days_used, 
    vector<vector<double>> apriori, 
    vector<vector<double>> doc_likelihood, 
    vector<vector<double>> cert_likelihood, 
    vector<vector<double>> days_used_mean, 
    vector<vector<double>> days_used_variance
);

// Signature methods to evalute model performance
vector<vector<double>> confusion_matrix(
    vector<vector<double>> bayes_result, 
    vector<vector<double>> test_data
);
vector<vector<double>> accuracy(
    vector<vector<double>> confusion_matrix
);


int main() {
    string file_path = "./data/dataset.csv";
    ifstream input_file;
    input_file.open(file_path); // Open the file
    // Check if the file is open
    if (!input_file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return 1; // Exit with error code
    }

    double id_val, doc_val_type, class_val, valid_certificated_val, days_used_val;
    vector<double> id;
    vector<double> doc_type;
    vector<double> class_vector;
    vector<double> valid_certificated;
    vector<double> days_used;

    string header; // Read the header line
    string cell; 

    getline(input_file, header); // Read the line and ignore it
    while (input_file.good()) 
    {
        getline(input_file, cell, ','); // Read the first column
        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());

        if (!cell.empty()) {

            id_val = stod(cell); // Convert to double
            id.push_back(id_val); // Store in vector
            
            getline(input_file, cell, ','); // Read the second column
            doc_val_type = stod(cell); // Convert to double
            doc_type.push_back(doc_val_type); // Store in vector

            getline(input_file, cell, ','); // Read the third column
            class_val = stod(cell); // Convert to double
            class_vector.push_back(class_val); // Store in vector

            getline(input_file, cell, ','); // Read the fourth column  
            valid_certificated_val = stod(cell); // Convert to double
            valid_certificated.push_back(valid_certificated_val); // Store in vector

            getline(input_file, cell, ','); // Read the fifth column
            days_used_val = stod(cell); // Convert to double
            days_used.push_back(days_used_val); // Store in vector

        } else {
            break; // Exit the loop if no more data
        }
    }

    auto start = high_resolution_clock::now(); // Start timing

    cout << "Starting the model..." << endl;

    // Split the data into training and testing sets
    cout << "... Splitting data into training and testing sets." << endl;

    int data_size = id.size();
    int TEST_SIZE = static_cast<int>(TEST_SIZE_PCT * data_size);

    vector<int> indexes(data_size);
    for (int i = 0; i < data_size; ++i) {
        indexes[i] = i;
    }
    // Shuffle the indexes
    random_device rd;
    mt19937 gen(rd());
    shuffle(indexes.begin(), indexes.end(), gen);

    vector<double> train_doc_type;
    vector<double> train_class_vector;
    vector<double> train_valid_certificated;
    vector<double> train_days_used;

    vector<double> test_doc_type;
    vector<double> test_class_vector;
    vector<double> test_valid_certificated;
    vector<double> test_days_used;

    for (int i = 0; i < data_size; ++i) {
        if (i < data_size - TEST_SIZE) {
            train_doc_type.push_back(doc_type[indexes[i]]);
            train_class_vector.push_back(class_vector[indexes[i]]);
            train_valid_certificated.push_back(valid_certificated[indexes[i]]);
            train_days_used.push_back(days_used[indexes[i]]);

        } else { 
            test_doc_type.push_back(doc_type[indexes[i]]);
            test_class_vector.push_back(class_vector[indexes[i]]);
            test_valid_certificated.push_back(valid_certificated[indexes[i]]);
            test_days_used.push_back(days_used[indexes[i]]);
        }
    }
    cout << "...... Train size: " << train_doc_type.size() << endl;
    cout << "...... Test size: " << test_doc_type.size() << endl;

    // Loop through the training data for the specified number of iterations
    // and print the details of each training sample for debugging purposes.
    for (int iteration = 0; iteration <= 20; iteration++) {
        cout << "Iteration: " << iteration << endl;
        cout << train_doc_type[iteration] << " " << 
                train_class_vector[iteration] << " " <<  
                train_valid_certificated[iteration] << " " <<  
                train_days_used[iteration] << " " <<  endl;

    }

    // Naive Bayes Classifier
    cout << "... Naive Bayes Classifier started." << endl;

    
    

}
