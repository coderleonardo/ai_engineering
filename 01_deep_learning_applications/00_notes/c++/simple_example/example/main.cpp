#include <iostream>
#include "author.h"

int main() {

   Author bob("Bob Silva", "bob@teste.com", 'm');
   bob.print();
   bob.set_email("bob@superteste.com");
   bob.print();

   std::cout << std::endl;
   Author mary("Mary Jane", "@super.com", 'f');
   mary.set_email("mary@mail.com");
   mary.print();

};