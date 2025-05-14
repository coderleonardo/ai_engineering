#include <string>

using namespace std;

class Author {

private:
    string name; 
    string email;
    char gender;

public: 
    // Constructor
    Author(const string& name, const string& email, char gender);

    // Getters
    string get_name() const;
    string get_email() const;
    string get_gender() const;

    // Setters
    void set_name(const string& name);
    void set_email(const string& email);

    void print() const;
};