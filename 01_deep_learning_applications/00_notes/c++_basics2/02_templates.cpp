/**
 * A template in C++ allows functions or classes to operate with generic types, enabling code reuse
 * for different data types without rewriting the same logic.
 */

#include <iostream>
using namespace std;

// Template function to find the maximum of two values
template <typename T>
T findMax(T a, T b) {
    return (a > b) ? a : b;
}

template <typename T>
T findMin(T a, T b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}


int main() {
    cout << "Max of 10 and 20: " << findMax(10, 20) << endl;
    cout << "Max of 5.5 and 2.3: " << findMax(5.5, 2.3) << endl;
    cout << "Max of 'A' and 'B': " << findMax('A', 'B') << endl;
    cout << endl;
    cout << "Min of 10 and 20: " << findMin(10, 20) << endl;
    cout << "Min of 5.5 and 2.3: " << findMin(5.5, 2.3) << endl;
    cout << "Min of 'A' and 'B': " << findMin('A', 'B') << endl;

    return 0;
}