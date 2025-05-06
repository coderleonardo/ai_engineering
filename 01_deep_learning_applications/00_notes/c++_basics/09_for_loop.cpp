#include <iostream>
using namespace std;

int main() {
    // Using a for loop to iterate from 0 to 4
    for (int i = 0; i < 5; i++) {
        cout << "Iteration: " << i << endl; // Print the current iteration number
    }
    
    // Using a for loop to iterate from 5 to 1 in reverse order
    for (int j = 5; j > 0; j--) {
        cout << "Reverse Iteration: " << j << endl; // Print the current reverse iteration number
    }

    for (int l = 9; l >= 0; l -= 2) {
        cout << "l: " << l << endl; // Print the current value of l
    }
    
    return 0; // Return 0 to indicate successful execution
} // End of main function