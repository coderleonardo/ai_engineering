// Livro.h - Declaração da classe Livro

#ifndef LIVRO_H
#define LIVRO_H

#include <string>
#include "Autor.h"

class Livro {
private:
    std::string nome;
    Autor autor;
    double valor;
    int quantidadeEstoque;

public:
    // Construtores
    Livro();
    Livro(const std::string& nome, const Autor& autor, double valor, int quantidadeEstoque);
    
    // Métodos de acesso (getters)
    std::string getNome() const;
    Autor getAutor() const;
    double getValor() const;
    int getQuantidadeEstoque() const;
    
    // Métodos de modificação (setters)
    void setNome(const std::string& nome);
    void setAutor(const Autor& autor);
    void setValor(double valor);
    void setQuantidadeEstoque(int quantidade);
};

#endif // LIVRO_H