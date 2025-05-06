#include <iostream>
using namespace std;

int main() {

    int arr1[5] = {1, 2, 3}; 
    int arr2[] = {1, 2, 3, 4, 5}; // Declare and initialize an array of integers with values 

    string arr3[5] = {"Hello", "World", "!"}; // Declare and initialize an array of strings with values

    for (int i = 0; i < 5; i++) {
        cout << "arr1[" << i << "] = " << arr1[i] << endl; // Print each element of the array
    }
    cout << "\n"; // Print a new line for better readability
    for (int i = 0; i < 5; i++) {
        cout << "arr2[" << i << "] = " << arr2[i] << endl; // Print each element of the array
    }
    cout << "\n"; // Print a new line for better readability
    for (int i = 0; i < 5; i++) {
        cout << "arr3[" << i << "] = " << arr3[i] << endl; // Print each element of the array
    }



}