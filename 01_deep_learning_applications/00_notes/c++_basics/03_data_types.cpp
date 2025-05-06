#include <iostream>
using namespace std; 

int main() {
    // Declare and initialize variables of different data types
    int a = 5; // Integer variable
    double b = 3.15; // Double variable

    int result_int = a + b; // Implicit conversion from double to int
    double result_double = a + b; // No conversion needed, both are double 

    cout << "Integer: " << a << endl; // Output integer value
    cout << "Double: " << b << endl; // Output double value
    cout << "Result (int): " << result_int << endl; // Output result of int addition  
    cout << "Result (double): " << result_double << endl; // Output result of double addition

    // When you dont declare a variable, it may contain garbage value
    int undeclared_variable;
    cout << "Undeclared variable: " << undeclared_variable << endl; // Output message

    return 0;

}