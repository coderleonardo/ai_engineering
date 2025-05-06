#include <iostream>
using namespace std;

int main() {

    int arr[5]; // Declare an array of integers with size 5
    arr[0] = 1; // Assign value 1 to the first element of the array
    arr[1] = 2; // Assign value 2 to the second element of the array
    arr[2] = 3; // Assign value 3 to the third element of the array
    arr[3] = 4; // Assign value 4 to the fourth element of the array
    arr[4] = 5; // Assign value 5 to the fifth element of the array
    // Print the values of the array
    for (int i = 0; i < 5; i++) {
        cout << "arr[" << i << "] = " << arr[i] << endl; // Print each element of the array
    }

    int arr2[5] = {1, 2, 3, 4, 5}; // Declare and initialize an array of integers with values
    for (int i = 0; i < 5; i++) {
        cout << "arr[" << i << "] = " << arr2[i] << endl; // Print each element of the array
    }

}