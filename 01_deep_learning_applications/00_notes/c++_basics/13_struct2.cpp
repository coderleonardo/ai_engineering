#include <iostream>
using namespace std;

struct Employee
{
    int id;
    string name;
    string position;
    double salary;
};

struct Company
{
    Employee CEO; 
    int numberOfEmployees; // Number of employees in the company
};

void printCompanyDetails(Company comp)
{
    cout << "CEO ID: " << comp.CEO.id << endl;
    cout << "CEO Name: " << comp.CEO.name << endl;
    cout << "CEO Position: " << comp.CEO.position << endl;
    cout << "CEO Salary: $" << comp.CEO.salary << endl;
    cout << "Number of Employees: " << comp.numberOfEmployees << endl;
};

int main()
{
    Company myCompany; // Declare a Company variable
    myCompany.CEO.id = 1; // Assign values to the CEO's attributes
    myCompany.CEO.name = "Alice Johnson";
    myCompany.CEO.position = "CEO";
    myCompany.CEO.salary = 120000.00;
    myCompany.numberOfEmployees = 50; // Assign the number of employees in the company
    cout << "Company Details:" << endl;
    printCompanyDetails(myCompany); // Call the function to print company details
    cout << endl;
    
    Company anotherCompany = {{2, "Bob Smith", "CTO", 110000.00}, 30}; // Declare and initialize another Company variable
    cout << "Another Company Details:" << endl;
    printCompanyDetails(anotherCompany); // Call the function to print company details
    cout << endl;

    return 0; // Return 0 to indicate successful execution
}
// End of main function
