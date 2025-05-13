#include <iostream>

int main() {
    // Example 1: Basic Pointer
    int a = 10;
    int* ptr = &a; // Pointer to the variable 'a'
    std::cout << "Value of a: " << a << std::endl;
    std::cout << "Address of a: " << &a << std::endl;
    std::cout << "Value stored in ptr (address of a): " << ptr << std::endl;
    std::cout << "Value pointed to by ptr: " << *ptr << std::endl; // Dereferencing the pointer to get the value of 'a'
    std::cout << std::endl;

    // Example 2: Pointer to an Array
    int arr[] = {1, 2, 3, 4, 5};
    int* arrPtr = arr; // Pointer to the first element of the array
    std::cout << "Address of arr: " << &arr << std::endl;
    std::cout << "Address of arrPtr: " << arrPtr << std::endl; // Address of the first element of the array
    std::cout << "\nArray elements using pointer:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << *(arrPtr + i) << " "; // Iterating through the array using pointer arithmetic
    }
    std::cout << std::endl;

    // Example 3: Null Pointer
    int* nullPtr = nullptr;
    std::cout << "\nNull pointer value: " << nullPtr << std::endl;
    std::cout << std::endl;

    // Example 4: Pointer to Pointer
    int** ptrToPtr = &ptr;
    std::cout << "\nValue of ptr: " << ptr << std::endl;
    std::cout << "Value of ptrToPtr (address of ptr): " << ptrToPtr << std::endl;
    std::cout << "Value pointed to by ptrToPtr (value of ptr): " << *ptrToPtr << std::endl;
    std::cout << "Value pointed to by the pointer pointed to by ptrToPtr: " << **ptrToPtr << std::endl;

    return 0;
}