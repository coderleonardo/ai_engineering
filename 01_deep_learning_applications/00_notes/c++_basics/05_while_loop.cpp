#include <iostream>
using namespace std;

int main() {

    cout << "Starting while loop..." << endl; // Output a message indicating the start of the while loop
    // The std::endl manipulator is used to insert a new line and flush the output buffer
    // The count object is used to output data to the console
    // The << operator is used to insert the data into the output stream
    int i = 0; // Initialize a variable i to 0
    while (i < 5) { // Loop while i is less than 5
        cout << "i: " << i << endl; // Output the current value of i
        i++; // Increment i by 1
    }
    cout << "While loop ended." << endl; // Output a message indicating the end of the while loop 

    cout << "Starting another while loop..." << endl; // Output a message indicating the start of another while loop
    int control1 = 1;
    while (control1 <= 5) {

        int control2 = 1;
        while (control2 <= control1) {
            cout << control2 << " "; // Output the current value of control2 followed by a space
            control2++; // Increment control2 by 1
        }
        cout << '\n'; // Output a new line
        control1++; 
        
    }

    return 0; // Return 0 to indicate successful execution
} // End of main function