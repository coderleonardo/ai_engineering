#include <iostream>
using namespace std;

int main() {
    int arr[5]; // Declare an array of integers with size 5

    for (int i = 0; i < 5; i++) {
        cout << "Enter number " << (i + 1) << ": "; // Prompt user for input
        cin >> arr[i]; // Read user input into the array
    }
    cout << "Sorting..." << endl; // Indicate sorting process
    // Bubble sort algorithm
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5 - i - 1; j++) { // Adjusted loop to avoid unnecessary comparisons
            if (arr[j] > arr[j + 1]) { // Compare adjacent elements
                // Swap elements if they are in the wrong order
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    cout << "Sorted array in ascending order:" << endl; // Indicate sorted order
    for (int i = 0; i < 5; i++) {
        cout << arr[i] << " "; // Print each element of the sorted array
    }
    cout << endl; // Print a new line for better readability
}