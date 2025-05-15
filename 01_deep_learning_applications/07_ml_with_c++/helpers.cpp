#include "helpers.h"
#include <iostream>
#include <cmath>
#include <numeric>

using namespace std;

// Function implementations
void print2d_vector(vector<vector<double>> vect) {
    for (int i = 0; i < vect.size(); i++) {
        for (int j = 0; j < vect[i].size(); j++) {
            cout << vect[i][j] << " ";
        }
        cout << endl;
    }
}

vector<vector<double>> prior_probability(vector<double> vect) {
    vector<vector<double>> result(1, vector<double>(2, 0));
    for (double val : vect) {
        if (val == 0) result[0][0]++;
        else if (val == 1) result[0][1]++;
    }
    if (!vect.empty()) {
        result[0][0] /= vect.size();
        result[0][1] /= vect.size();
    }
    return result;
}

vector<vector<double>> count_classes(vector<double> vect) {
    vector<vector<double>> result(2, vector<double>(2, 0));
    for (double val : vect) {
        if (val == 0) result[0][0]++;
        else if (val == 1) result[1][1]++;
    }
    return result;
}

vector<vector<double>> document_likelihood(vector<double> class_vector, vector<double> doc_type, vector<vector<double>> class_count) {
    vector<vector<double>> result(2, vector<double>(3, 0));
    for (int i = 0; i < class_vector.size(); i++) {
        if (class_vector[i] == 1) result[0][static_cast<int>(doc_type[i])]++;
        else if (class_vector[i] == 0) result[1][static_cast<int>(doc_type[i])]++;
    }
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) result[i][j] /= class_count[i][j];
        }
    }
    return result;
}

vector<vector<double>> certificated_likelihood(vector<double> class_vector, vector<double> valid_certificated, vector<vector<double>> class_count) {
    vector<vector<double>> result(2, vector<double>(2, 0));
    for (int i = 0; i < class_vector.size(); i++) {
        result[static_cast<int>(class_vector[i])][static_cast<int>(valid_certificated[i])]++;
    }
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < result[i].size(); j++) {
            if (class_count[i][j] != 0) result[i][j] /= class_count[i][j];
        }
    }
    return result;
}

vector<vector<double>> days_used_mean(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count) {
    vector<vector<double>> result(1, vector<double>(2, 0));
    for (int i = 0; i < class_vector.size(); i++) {
        result[0][static_cast<int>(class_vector[i])] += days_used[i];
    }
    for (int j = 0; j < result[0].size(); j++) {
        if (class_count[0][j] != 0) result[0][j] /= class_count[0][j];
    }
    return result;
}

vector<vector<double>> days_used_variance(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count) {
    vector<vector<double>> result(1, vector<double>(2, 0));
    for (int i = 0; i < class_vector.size(); i++) {
        result[0][static_cast<int>(class_vector[i])] += pow(days_used[i], 2);
    }
    for (int j = 0; j < result[0].size(); j++) {
        if (class_count[0][j] != 0) result[0][j] /= class_count[0][j];
    }
    return result;
}

vector<vector<double>> days_used_metrics(vector<vector<double>> days_used_mean, vector<vector<double>> days_used_variance) {
    vector<vector<double>> result(2, vector<double>(2, 0));
    result[0][0] = days_used_mean[0][0];
    result[0][1] = sqrt(days_used_variance[0][0]);
    result[1][0] = days_used_mean[0][1];
    result[1][1] = sqrt(days_used_variance[0][1]);
    return result;
}

double calculate_days_used(double v, double mean_v, double variance_v) {
    if (variance_v > 0) {
        return (1 / sqrt(2 * M_PI * variance_v)) * exp(-pow(v - mean_v, 2) / (2 * variance_v));
    }
    return 0.0;
}

vector<vector<double>> bayes_theorem(double doc_type, double valid_certificated, double days_used, vector<vector<double>> apriori, vector<vector<double>> doc_likelihood, vector<vector<double>> cert_likelihood, vector<vector<double>> days_used_mean, vector<vector<double>> days_used_variance) {
    vector<vector<double>> result(1, vector<double>(2, 0));
    result[0][0] = apriori[0][0] * doc_likelihood[0][static_cast<int>(doc_type)] * cert_likelihood[0][static_cast<int>(valid_certificated)] * calculate_days_used(days_used, days_used_mean[0][0], days_used_variance[0][0]);
    result[0][1] = apriori[0][1] * doc_likelihood[1][static_cast<int>(doc_type)] * cert_likelihood[1][static_cast<int>(valid_certificated)] * calculate_days_used(days_used, days_used_mean[0][1], days_used_variance[0][1]);
    double sum = accumulate(result[0].begin(), result[0].end(), 0.0);
    if (sum != 0) {
        for (double &val : result[0]) val /= sum;
    }
    return result;
}

vector<vector<double>> confusion_matrix(vector<double> bayes_result, vector<double> test_data) {
    vector<vector<double>> result(2, vector<double>(2, 0));
    for (int i = 0; i < bayes_result.size(); i++) {
        result[static_cast<int>(bayes_result[i])][static_cast<int>(test_data[i])]++;
    }
    return result;
}

vector<vector<double>> accuracy(vector<vector<double>> confusion_matrix) {
    vector<vector<double>> result(1, vector<double>(2, 0));
    double TP = confusion_matrix[0][0];
    double TN = confusion_matrix[1][1];
    double FP = confusion_matrix[1][0];
    double FN = confusion_matrix[0][1];
    double total = TP + TN + FP + FN;
    if (total != 0) {
        result[0][0] = (TP + TN) / total;
        result[0][1] = TP / (TP + FN);
    }
    return result;
}