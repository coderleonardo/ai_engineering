// Livro.cpp - Implementação da classe Livro

#include "Livro.h"

// Construtor padrão
Livro::Livro() : nome(""), autor(), valor(0.0), quantidadeEstoque(0) {
}

// Construtor com parâmetros
Livro::Livro(const std::string& nome, const Autor& autor, double valor, int quantidadeEstoque)
    : nome(nome), autor(autor), valor(valor), quantidadeEstoque(quantidadeEstoque) {
}

// Implementação dos getters
std::string Livro::getNome() const {
    return nome;
}

Autor Livro::getAutor() const {
    return autor;
}

double Livro::getValor() const {
    return valor;
}

int Livro::getQuantidadeEstoque() const {
    return quantidadeEstoque;
}

// Implementação dos setters
void Livro::setNome(const std::string& nome) {
    this->nome = nome;
}

void Livro::setAutor(const Autor& autor) {
    this->autor = autor;
}

void Livro::setValor(double valor) {
    this->valor = valor;
}

void Livro::setQuantidadeEstoque(int quantidade) {
    this->quantidadeEstoque = quantidade;
}