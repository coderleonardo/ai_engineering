#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "helpers.h"

using namespace std;
using namespace std::chrono;

const double TEST_SIZE_PCT = 0.3;
const int NUM_ITERATIONS = 5;

int main() {
    string file_path = "./data/dataset.csv";
    ifstream input_file(file_path);
    if (!input_file.is_open()) {
        cerr << "Error opening file: " << file_path << endl;
        return 1;
    }

    vector<double> id, doc_type, class_vector, valid_certificated, days_used;
    string header, cell;
    getline(input_file, header);
    while (getline(input_file, cell, ',')) {
        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());
        if (!cell.empty()) {
            id.push_back(stod(cell));
            getline(input_file, cell, ','); doc_type.push_back(stod(cell));
            getline(input_file, cell, ','); class_vector.push_back(stod(cell));
            getline(input_file, cell, ','); valid_certificated.push_back(stod(cell));
            getline(input_file, cell, ','); days_used.push_back(stod(cell));
        }
    }

    auto start = high_resolution_clock::now();
    cout << "Starting the model..." << endl;

    int data_size = id.size();
    int TEST_SIZE = static_cast<int>(TEST_SIZE_PCT * data_size);
    vector<int> indexes(data_size);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), mt19937(random_device()()));

    vector<double> train_doc_type, train_class_vector, train_valid_certificated, train_days_used;
    vector<double> test_doc_type, test_class_vector, test_valid_certificated, test_days_used;

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

    cout << "... Naive Bayes Classifier started." << endl;
    vector<vector<double>> apriori = prior_probability(train_class_vector);
    vector<vector<double>> class_count = count_classes(train_class_vector);
    vector<vector<double>> doc_likelihood = document_likelihood(train_class_vector, train_doc_type, class_count);
    vector<vector<double>> cert_likelihood = certificated_likelihood(train_class_vector, train_valid_certificated, class_count);
    vector<vector<double>> days_used_avg = days_used_mean(train_class_vector, train_days_used, class_count);
    vector<vector<double>> days_used_var = days_used_variance(train_class_vector, train_days_used, class_count);

    print2d_vector(apriori);
    print2d_vector(class_count);
    print2d_vector(doc_likelihood);
    print2d_vector(cert_likelihood);

    vector<vector<double>> days_used_metrics_result = days_used_metrics(days_used_avg, days_used_var);
    print2d_vector(days_used_metrics_result);

    auto stop = high_resolution_clock::now();
    vector<vector<double>> raw(1, vector<double>(2, 0));

    cout << "... Classifying test data." << endl;
    vector<double> probabilities(test_doc_type.size());
    for (int i = 0; i < test_doc_type.size(); i++) {
        raw = bayes_theorem(
            test_doc_type[i], 
            test_valid_certificated[i], 
            test_days_used[i], 
            apriori, 
            doc_likelihood, 
            cert_likelihood, 
            days_used_avg, 
            days_used_var
        );
        probabilities[i] = (raw[0][0] > raw[0][1]) ? 0 : 1;
    }

    cout << "... Confusion matrix." << endl;
    vector<vector<double>> confusion_matrix_result = confusion_matrix(probabilities, test_class_vector);
    print2d_vector(confusion_matrix_result);

    cout << "... Accuracy." << endl;
    vector<vector<double>> accuracy_result = accuracy(confusion_matrix_result);
    print2d_vector(accuracy_result);

    chrono::duration<double> elapsed = stop - start;
    cout << "Time elapsed: " << elapsed.count() << " seconds." << endl;

    return 0;
}