#include <iostream>

int main() {
    
    int x(5); // Declare and initialize an integer variable x with the value 5

    std::cout << "The value of x is: " << x << std::endl; // Output the value of x to the console
    // The std::endl manipulator is used to insert a new line and flush the output buffer

    std::cout << "\nType a value for y: "; // Prompt the user to enter a value for y
    int y; // Declare an integer variable y
    std::cin >> y; // Read the value of y from the console input
    // The std::cin object is used to read input from the standard input stream (keyboard)
    // The >> operator is used to extract the value from the input stream and store it in the variable y
    std::cout << "The value of y is: " << y << std::endl; // Output the value of y to the console
    // The std::endl manipulator is used to insert a new line and flush the output buffer

    return 0; // Return 0 to indicate successful execution
} // End of main function