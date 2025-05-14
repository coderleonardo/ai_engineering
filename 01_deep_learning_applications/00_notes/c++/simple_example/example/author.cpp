#include <iostream>
#include "author.h"

using namespace std;


// Specifying the constructor
Author::Author(const string& name, const string& email, char gender) {

    set_name(name);
    set_email(email);

    if (gender == 'm' || gender == 'f') {
        this -> gender = gender;
    }
    else {
        cout << "Invalid gender!" << endl;
    }
};

// Specifying the methods getter and setter
string Author::get_name() const {
    return name;
}; 
string Author::get_email() const {
    return email;
};

void Author::set_name(const string& name) {
    this -> name = name;
};
void Author::set_email(const string& email) {
    this -> email = email;
}; 

void Author::print() const {
    cout << "Name: " << name << endl;
    cout << "Email: " << email << endl;
    cout << "Gender: " << gender << endl;
}; 

