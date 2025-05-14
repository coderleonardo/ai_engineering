// main.cpp - Arquivo de teste para as classes Autor e Livro

#include <iostream>
#include <iomanip>
#include "Autor.h"
#include "Livro.h"

// Função auxiliar para exibir informações completas de um livro
void exibirInformacoesLivro(const Livro& livro) {
    std::cout << "================================\n";
    std::cout << "Informações do Livro:\n";
    std::cout << "--------------------------------\n";
    std::cout << "Nome: " << livro.getNome() << std::endl;
    std::cout << "Valor: R$ " << std::fixed << std::setprecision(2) << livro.getValor() << std::endl;
    std::cout << "Estoque: " << livro.getQuantidadeEstoque() << " unidades\n";
    
    // Informações do autor
    Autor autor = livro.getAutor();
    std::cout << "--------------------------------\n";
    std::cout << "Informações do Autor:\n";
    std::cout << "Nome: " << autor.getNome() << std::endl;
    std::cout << "Email: " << autor.getEmail() << std::endl;
    std::cout << "Gênero: " << autor.getGenero() << " (M=Masculino, F=Feminino, O=Outro)\n";
    std::cout << "================================\n\n";
}

int main() {
    std::cout << "Teste das Classes Livro e Autor\n\n";
    
    // Teste 1: Criando um autor e um livro com construtores padrão e depois configurando
    std::cout << "Teste 1: Usando construtores padrão e setters\n";
    Autor autor1;
    autor1.setNome("Machado de Assis");
    autor1.setEmail("machado@literatura.br");
    autor1.setGenero('M');
    
    Livro livro1;
    livro1.setNome("Dom Casmurro");
    livro1.setAutor(autor1);
    livro1.setValor(29.90);
    livro1.setQuantidadeEstoque(50);
    
    exibirInformacoesLivro(livro1);
    
    // Teste 2: Usando construtores com parâmetros
    std::cout << "Teste 2: Usando construtores com parâmetros\n";
    Autor autor2("Clarice Lispector", "clarice@literatura.br", 'F');
    Livro livro2("A Hora da Estrela", autor2, 24.50, 30);
    
    exibirInformacoesLivro(livro2);
    
    // Teste 3: Modificando valores
    std::cout << "Teste 3: Modificando valores de um livro existente\n";
    std::cout << "Estado original:\n";
    exibirInformacoesLivro(livro2);
    
    // Alterando valores
    livro2.setValor(27.90);
    livro2.setQuantidadeEstoque(15);
    
    // Atualizando o email do autor
    Autor autorAtualizado = livro2.getAutor();
    autorAtualizado.setEmail("clarice.lispector@editorial.com");
    livro2.setAutor(autorAtualizado);
    
    std::cout << "Após modificações:\n";
    exibirInformacoesLivro(livro2);
    
    return 0;
}