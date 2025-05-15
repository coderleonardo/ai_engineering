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
    vector<double> bayes_result, 
    vector<double> test_data
);
vector<vector<double>> accuracy(
    vector<vector<double>> confusion_matrix
);


int main() {
    string file_path = "./data/dataset.csv";
    std::ifstream input_file;
    input_file.open(file_path); // Open the file
    // Check if the file is open
    if (!input_file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
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

    std::cout << "Starting the model..." << std::endl;

    // Split the data into training and testing sets
    cout << "... Splitting data into training and testing sets." << endl;

    int data_size = id.size();
    int TEST_SIZE = static_cast<int>(TEST_SIZE_PCT * data_size);

    vector<int> indexes(data_size);
    for (int i = 0; i < data_size; ++i) {
        indexes[i] = i;
    }
    // Shuffle the indexes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indexes.begin(), indexes.end(), gen);

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

    
    // apriori probability
    vector<vector<double>> apriori = prior_probability(train_class_vector);

    // class count
    vector<vector<double>> class_count = count_classes(train_class_vector);

    // document likelihood
    vector<vector<double>> doc_likelihood = document_likelihood(train_class_vector, train_doc_type, class_count);
    // certificated likelihood
    vector<vector<double>> cert_likelihood = certificated_likelihood(train_class_vector, train_valid_certificated, class_count);
    // days used mean
    vector<vector<double>> days_used_avg = days_used_mean(train_class_vector, train_days_used, class_count);
    // days used variance
    vector<vector<double>> days_used_var = days_used_variance(train_class_vector, train_days_used, class_count);

    print2d_vector(apriori);
    print2d_vector(class_count);    
    print2d_vector(doc_likelihood);
    print2d_vector(cert_likelihood);

    vector<vector<double>> days_used_metrics_result = days_used_metrics(days_used_avg, days_used_var);
    print2d_vector(days_used_metrics_result);

    auto stop = high_resolution_clock::now(); // Stop timing

    vector<vector<double>> raw(1, vector<double>(2, 0)); // Initialize raw with 1 row and 2 columns

    cout << "... Classifying test data." << endl;
    for (int i = 0; i < (TEST_SIZE + NUM_ITERATIONS); i++) {
        // Classify each test data point
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

        print2d_vector(raw);
    }

    // get time elapsed
    std::chrono::duration<double> elapsed = stop - start;
    cout << "Time elapsed: " << elapsed.count() << " seconds." << endl;

    // Normalize the probabilities
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

        if (raw[0][0] > 0.5) {
            probabilities[i] = 0;
        } else if (raw[0][1] > 0.5) {
            probabilities[i] = 1;
        } else {}
    }

    // Calculate and show metrics
    cout << "... Confusion matrix." << endl;
    vector<vector<double>> confusion_matrix_result = confusion_matrix(probabilities, test_class_vector);
    print2d_vector(confusion_matrix_result);
    cout << "... Accuracy." << endl;
    vector<vector<double>> accuracy_result = accuracy(confusion_matrix_result);

    return 0;

}

// Show 2D vector
void print2d_vector(vector<vector<double>> vect) {
    for (int i = 0; i < vect.size(); i++) {
        for (int j = 0; j < vect[i].size(); j++) {
            std::cout << vect[i][j] << " ";
        }
        cout << endl;   
    }
};

// Calculate the apriori probability
vector<vector<double>> prior_probability(vector<double> vect) {
    vector<vector<double>> result(1, vector<double>(2, 0)); // Initialize result with 1 row and 2 columns
    
    for (int i = 0; i < vect.size(); i++) {
        if (vect[i] == 0) {
            result[0][0]++;
        } else if (vect[i] == 1) {
            result[0][1]++;
        }
    }
    if (vect.size() > 0) {
        result[0][0] = result[0][0] / vect.size();
        result[0][1] = result[0][1] / vect.size();
    } else {
        result[0][0] = 0;
        result[0][1] = 0;
    }
    return result;
};
// Count the number of classes
vector<vector<double>> count_classes(vector<double> vect) {
    vector<vector<double>> result(2, vector<double>(2, 0)); // Initialize result with 2 rows and 2 columns
    
    for (int i = 0; i < vect.size(); i++) {
        if (vect[i] == 0) {
            result[0][0]++;
        } else if (vect[i] == 1) {
            result[1][1]++;
        }
    }
    return result;
};
// Calculate the document likelihood
vector<vector<double>> document_likelihood(vector<double> class_vector, vector<double> doc_type, vector<vector<double>> class_count) {
    vector<vector<double>> result(2, vector<double>(3, 0)); // Initialize result with 2 rows and 3 columns

    for (int i = 0; i < class_vector.size(); i++) {
        if (class_vector[i] == 1) {
            if (doc_type[i] == 0) {
                result[0][0]++;
            } else if (doc_type[i] == 2) {
                result[0][1]++;
            } else if (doc_type[i] == 3) {
                result[0][2]++;
            } else {}
        } else if (class_vector[i] == 0) {
            if (doc_type[i] == 1) {
                result[1][0]++;
            } else if (doc_type[i] == 2) {
                result[1][1]++;
            } else if (doc_type[i] == 3) {
                result[1][2]++;
            } else {}
        } else {}
    }
    // Normalize the result
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) {
                result[i][j] = result[i][j] / class_count[i][j];
            } else {
                result[i][j] = 0;
            }
        }
    }
    return result;
};
// Calculate the certificated likelihood
vector<vector<double>> certificated_likelihood(vector<double> class_vector, vector<double> valid_certificated, vector<vector<double>> class_count) {
    vector<vector<double>> result(2, vector<double>(2, 0)); // Initialize result with 2 rows and 2 columns

    for (int i = 0; i < class_vector.size(); i++) {
        if (class_vector[i] == 0) {
            if (valid_certificated[i] == 0) {
                result[0][0]++;
            } else if (valid_certificated[i] == 1) {
                result[0][1]++;
            } else {}
        } else if (class_vector[i] == 1) {
            if (valid_certificated[i] == 0) {
                result[1][0]++;
            } else if (valid_certificated[i] == 1) {
                result[1][1]++;
            } else {}
        } else {}
    }
    // Normalize the result
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) {
                result[i][j] = result[i][j] / class_count[i][j];
            } else {
                result[i][j] = 0;
            }
        }
    }
    return result;
};
// Calculate the days used mean
vector<vector<double>> days_used_mean(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count) {
    vector<vector<double>> result(1, vector<double>(2, 0)); // Initialize result with 1 row and 2 columns
    
    for (int i = 0; i < class_vector.size(); i++) {
        if (class_vector[i] == 0) {
            result[0][0] += days_used[i];
        } else if (class_vector[i] == 1) {
            result[0][1] += days_used[i];
        } else {}
    }
    // Normalize the result
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) {
                result[i][j] = result[i][j] / class_count[i][j];
            } else {
                result[i][j] = 0;
            }
        }
    }
    return result;
};
// Calculate the days used variance
vector<vector<double>> days_used_variance(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count) {
    vector<vector<double>> result(1, vector<double>(2, 0)); // Initialize result with 1 row and 2 columns
    
    for (int i = 0; i < class_vector.size(); i++) {
        if (class_vector[i] == 0) {
            result[0][0] += pow(days_used[i], 2);
        } else if (class_vector[i] == 1) {
            result[0][1] += pow(days_used[i], 2);
        } else {}
    }
    // Normalize the result
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) {
                result[i][j] = result[i][j] / class_count[i][j];
            } else {
                result[i][j] = 0;
            }
        }
    }
    return result;
};
// Calculate the days used metrics
vector<vector<double>> days_used_metrics(vector<vector<double>> days_used_mean, vector<vector<double>> days_used_variance) {
    vector<vector<double>> result(2, vector<double>(2, 0)); // Initialize result with 2 rows and 2 columns

    result[0][0] = days_used_mean[0][0];
    result[0][1] = sqrt(days_used_variance[0][0]);
    result[1][0] = days_used_mean[0][1];
    result[1][1] = sqrt(days_used_variance[0][1]);

    return result;
};
// Calculate days used probability
double calculate_days_used(double v, double mean_v, double variance_v) {
    if (variance_v > 0) {
        double result = (1 / (sqrt(2 * M_PI * variance_v))) * exp(-pow(v - mean_v, 2) / (2 * variance_v));
        return result;
    } else {
        return 0.0; // Return 0 or an appropriate default value if variance_v is zero
    }
};
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
) {
    vector<vector<double>> result(1, vector<double>(2, 0)); // Initialize result with 1 row and 2 columns

    // Populate the result vector with meaningful values
    result[0][0] = apriori[0][0] * doc_likelihood[0][static_cast<int>(doc_type)] * cert_likelihood[0][static_cast<int>(valid_certificated)] * calculate_days_used(days_used, days_used_mean[0][0], days_used_variance[0][0]);
    result[0][1] = apriori[0][1] * doc_likelihood[1][static_cast<int>(doc_type)] * cert_likelihood[1][static_cast<int>(valid_certificated)] * calculate_days_used(days_used, days_used_mean[0][1], days_used_variance[0][1]);

    // Normalize the result
    double sum = accumulate(result[0].begin(), result[0].end(), 0.0);
    if (sum != 0) {
        for (int i = 0; i < result[0].size(); i++) {
            result[0][i] = 0; // Avoid division by zero
        }
    } else {
        for (int i = 0; i < result[0].size(); i++) {
            result[0][i] = 0; // Assign 0 if sum is zero to avoid division by zero
        }
    }
    // Normalize the result
    return result;
};
// Calculate the confusion matrix
vector<vector<double>> confusion_matrix(
    vector<double> bayes_result, 
    vector<double> test_data
) {
    vector<vector<double>> result(2, vector<double>(2, 0)); // Initialize result with 2 rows and 2 columns

    for (int i = 0; i < bayes_result.size(); i++) {
        if (bayes_result[i] == 0 && test_data[i] == 0) {
            result[0][0]++;
        } else if (bayes_result[i] == 1 && test_data[i] == 1) {
            result[1][1]++;
        } else if (bayes_result[i] == 0 && test_data[i] == 1) {
            result[0][1]++;
        } else if (bayes_result[i] == 1 && test_data[i] == 0) {
            result[1][0]++;
        } else {}
    }
    return result;
};
// Calculate the accuracy
vector<vector<double>> accuracy(
    vector<vector<double>> confusion_matrix
) {
    vector<vector<double>> result(1, vector<double>(2, 0)); // Initialize result with 1 row and 2 columns

    double TP = confusion_matrix[0][0];
    double TN = confusion_matrix[1][1];
    double FP = confusion_matrix[1][0];
    double FN = confusion_matrix[0][1];

    if ((TP + TN + FP + FN) != 0) {
        result[0][0] = (TP + TN) / (TP + TN + FP + FN);
    } else {
        result[0][0] = 0; // Handle division by zero
    }

    result[0][0] = (TP + TN) / (TP + TN + FP + FN);
    result[0][1] = (TP) / (TP + FN);

    return result;
}