#include <iostream>
using namespace std;

int main() {

    int x, y;
    int operation; // Variable to store the operation
    do {
        cout << "Enter a number for the  x variable: ";
        cin >> x; // Read user input for x
        cout << "Enter a number for the  y variable: "; 
        cin >> y; // Read user input for y

        cout << "Enter with the symbol of the operation you want to do: " << endl;
        cout << "1. Addition" << endl; // Option for addition
        cout << "2. Subtraction" << endl; // Option for subtraction
        cout << "3. Multiplication" << endl; // Option for multiplication
        cout << "4. Division" << endl; // Option for division
        
        cin >> operation; // Read user input for the operation

        switch (operation) { // Switch statement to perform the selected operation
            case 1: // Addition
                cout << "Result: " << x + y << endl; // Print the result of addition
                break; // Exit the switch statement
            case 2: // Subtraction
                cout << "Result: " << x - y << endl; // Print the result of subtraction
                break; // Exit the switch statement
            case 3: // Multiplication
                cout << "Result: " << x * y << endl; // Print the result of multiplication
                break; // Exit the switch statement
            case 4: // Division
                if (y != 0) { // Check if y is not zero to avoid division by zero
                    cout << "Result: " << x / y << endl; // Print the result of division
                } else {
                    cout << "Error: Division by zero is not allowed." << endl; // Print error message for division by zero
                }
                break; // Exit the switch statement
            default: // Invalid operation
                cout << "Invalid operation. Please try again." << endl; // Print error message for invalid operation
        }
    } while (operation < 1 || operation > 4); // Continue until a valid operation is entered
}