#include <iostream>
#include <string>

using namespace std;

enum Color
{
    RED,
    GREEN,
    BLUE
};
enum class Fruit
{
    APPLE,
    BANANA,
    ORANGE
};
enum class Animal
{
    DOG,
    CAT,
    BIRD
};

string get_color_name(Color color)
{
    switch (color)
    {
    case RED:
        return "Red";
    case GREEN:
        return "Green";
    case BLUE:
        return "Blue";
    default:
        return "Unknown Color";
    }
}

string get_fruit_name(Fruit fruit)
{
    if (fruit == Fruit::APPLE)
        return "Apple";
    else if (fruit == Fruit::BANANA)
        return "Banana";
    else if (fruit == Fruit::ORANGE)
        return "Orange";
    else
        return "Unknown Fruit";
}

string get_animal_name(Animal animal)
{
    if (animal == Animal::DOG) 
        return "Dog";
    else if (animal == Animal::CAT)
        return "Cat";
    else if (animal == Animal::BIRD)
        return "Bird";
    else
        return "Unknown Animal";
}

int main()
{
    Color color = RED;
    Fruit fruit = Fruit::BANANA;
    Animal animal = Animal::CAT;

    cout << "Color: " << get_color_name(color) << endl;
    cout << "Fruit: " << get_fruit_name(fruit) << endl;
    cout << "Animal: " << get_animal_name(animal) << endl;

    return 0;
}