#include <iostream>
using namespace std;

struct employee
{
    int id;
    string name;
    string position;
    double salary;
};

void printEmployeeDetails(employee emp)
{
    cout << "ID: " << emp.id << endl;
    cout << "Name: " << emp.name << endl;
    cout << "Position: " << emp.position << endl;
    cout << "Salary: $" << emp.salary << endl;
};

int main()
{
    employee emp1; // Declare an employee variable
    emp1.id = 1; // Assign values to the employee's attributes
    emp1.name = "John Doe";
    emp1.position = "Software Engineer";
    emp1.salary = 75000.00;

    cout << "Employee Details:" << endl;
    printEmployeeDetails(emp1); // Call the function to print employee details

    cout << endl; 

    employee emp2 = {2, "Jane Smith", "Project Manager", 85000.00}; // Declare and initialize another employee variable

    cout << "Employee Details:" << endl;
    printEmployeeDetails(emp2); // Call the function to print employee details

    return 0; // Return 0 to indicate successful execution
}
// End of main function
