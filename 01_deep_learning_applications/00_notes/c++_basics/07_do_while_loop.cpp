#include <iostream>
using namespace std;

int main() {
    int selector; 

    do {
        cout << "Enter a positive number: ";
        cin >> selector; // Read user input
    }
    while (selector < 0);
    cout << "You entered: " << selector << endl; // Output the entered number

    int i = 1;
    do {
        cout << i << " "; // Output the current value of i followed by a space
        i++;
    } while (i <= selector); // Increment i and check if it's less than 5
    cout << endl; // Output a new line
        

}