#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <string>

using namespace std;

// Function declarations
void print2d_vector(vector<vector<double>> vect);
vector<vector<double>> prior_probability(vector<double> vect);
vector<vector<double>> count_classes(vector<double> vect);
vector<vector<double>> document_likelihood(vector<double> class_vector, vector<double> doc_type, vector<vector<double>> class_count);
vector<vector<double>> certificated_likelihood(vector<double> class_vector, vector<double> valid_certificated, vector<vector<double>> class_count);
vector<vector<double>> days_used_mean(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count);
vector<vector<double>> days_used_variance(vector<double> class_vector, vector<double> days_used, vector<vector<double>> class_count);
vector<vector<double>> days_used_metrics(vector<vector<double>> days_used_mean, vector<vector<double>> days_used_variance);
double calculate_days_used(double v, double mean_v, double variance_v);
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
vector<vector<double>> confusion_matrix(vector<double> bayes_result, vector<double> test_data);
vector<vector<double>> accuracy(vector<vector<double>> confusion_matrix);

#endif // HELPERS_H