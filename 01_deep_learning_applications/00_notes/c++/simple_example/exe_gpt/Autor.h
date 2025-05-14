// Autor.h - Declaração da classe Autor

#ifndef AUTOR_H
#define AUTOR_H

#include <string>

class Autor {
private:
    std::string nome;
    std::string email;
    char genero;

public:
    // Construtor
    Autor();
    Autor(const std::string& nome, const std::string& email, char genero);
    
    // Métodos de acesso (getters)
    std::string getNome() const;
    std::string getEmail() const;
    char getGenero() const;
    
    // Métodos de modificação (setters)
    void setNome(const std::string& nome);
    void setEmail(const std::string& email);
    void setGenero(char genero);
};

#endif // AUTOR_H