// Autor.cpp - Implementação da classe Autor

#include "Autor.h"

// Construtor padrão
Autor::Autor() : nome(""), email(""), genero(' ') {
}

// Construtor com parâmetros
Autor::Autor(const std::string& nome, const std::string& email, char genero) 
    : nome(nome), email(email), genero(genero) {
}

// Implementação dos getters
std::string Autor::getNome() const {
    return nome;
}

std::string Autor::getEmail() const {
    return email;
}

char Autor::getGenero() const {
    return genero;
}

// Implementação dos setters
void Autor::setNome(const std::string& nome) {
    this->nome = nome;
}

void Autor::setEmail(const std::string& email) {
    this->email = email;
}

void Autor::setGenero(char genero) {
    this->genero = genero;
}