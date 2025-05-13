#include <iostream>
#include <string>
using namespace std;

// Classe BST (Árvore Binária de Busca)
class BinarySearchTree {
private:
    // Estrutura de nó interno
    struct Node {
        int data;           // Valor armazenado no nó
        Node* left;         // Ponteiro para subárvore esquerda
        Node* right;        // Ponteiro para subárvore direita
        
        // Construtor do nó
        Node(int value) : data(value), left(nullptr), right(nullptr) {}
    };
    
    // Raiz da árvore
    Node* root;
    
    // Métodos auxiliares privados (implementados abaixo)
    Node* insertRecursive(Node* node, int value);
    Node* findMinNode(Node* node);
    Node* removeRecursive(Node* node, int value);
    void printInOrderRecursive(Node* node);
    void destroyRecursive(Node* node);
    Node* searchRecursive(Node* node, int value);
    
public:
    // Construtor da árvore - inicializa a raiz como nula
    BinarySearchTree() : root(nullptr) {}
    
    // Destrutor da árvore - libera toda memória alocada
    ~BinarySearchTree() {
        destroyRecursive(root);
    }
    
    // Métodos públicos
    void insert(int value) {
        root = insertRecursive(root, value);
    }
    
    bool search(int value) {
        return searchRecursive(root, value) != nullptr;
    }
    
    void remove(int value) {
        root = removeRecursive(root, value);
    }
    
    void printInOrder() {
        printInOrderRecursive(root);
        cout << endl;
    }
    
    bool isEmpty() {
        return root == nullptr;
    }
};

// Implementações dos métodos privados

// Inserção recursiva
BinarySearchTree::Node* BinarySearchTree::insertRecursive(Node* node, int value) {
    // Caso base: chegamos a um ponto de inserção
    if (node == nullptr) {
        return new Node(value);
    }
    
    // Decide para qual lado seguir com base no valor
    if (value < node->data) {
        node->left = insertRecursive(node->left, value);
    } else if (value > node->data) {
        node->right = insertRecursive(node->right, value);
    }
    // Se o valor já existe, não faz nada (nenhuma duplicata)
    
    return node;
}

// Busca recursiva
BinarySearchTree::Node* BinarySearchTree::searchRecursive(Node* node, int value) {
    // Caso base: nó nulo (não encontrado) ou valor encontrado
    if (node == nullptr || node->data == value) {
        return node;
    }
    
    // Busca à esquerda ou à direita conforme o valor
    if (value < node->data) {
        return searchRecursive(node->left, value);
    } else {
        return searchRecursive(node->right, value);
    }
}

// Encontra o nó com o menor valor na subárvore
BinarySearchTree::Node* BinarySearchTree::findMinNode(Node* node) {
    // O valor mínimo está sempre no nó mais à esquerda
    while (node != nullptr && node->left != nullptr) {
        node = node->left;
    }
    return node;
}

// Remoção recursiva
BinarySearchTree::Node* BinarySearchTree::removeRecursive(Node* node, int value) {
    // Caso base: árvore vazia ou nó não encontrado
    if (node == nullptr) {
        return nullptr;
    }
    
    // Busca o nó a ser removido
    if (value < node->data) {
        // O valor está na subárvore esquerda
        node->left = removeRecursive(node->left, value);
    } else if (value > node->data) {
        // O valor está na subárvore direita
        node->right = removeRecursive(node->right, value);
    } else {
        // Encontramos o nó a ser removido!
        
        // Caso 1: Nó folha (sem filhos)
        if (node->left == nullptr && node->right == nullptr) {
            delete node;
            return nullptr;
        }
        // Caso 2: Nó com apenas um filho
        else if (node->left == nullptr) {
            Node* temp = node->right;
            delete node;
            return temp;
        }
        else if (node->right == nullptr) {
            Node* temp = node->left;
            delete node;
            return temp;
        }
        // Caso 3: Nó com dois filhos
        else {
            // Encontra o sucessor (menor valor na subárvore direita)
            Node* temp = findMinNode(node->right);
            
            // Copia o valor do sucessor para este nó
            node->data = temp->data;
            
            // Remove o sucessor
            node->right = removeRecursive(node->right, temp->data);
        }
    }
    
    return node;
}

// Percurso em ordem (in-order traversal)
void BinarySearchTree::printInOrderRecursive(Node* node) {
    if (node != nullptr) {
        // Esquerda, Raiz, Direita
        printInOrderRecursive(node->left);
        cout << node->data << " ";
        printInOrderRecursive(node->right);
    }
}

// Destrói a árvore recursivamente
void BinarySearchTree::destroyRecursive(Node* node) {
    if (node != nullptr) {
        // Libera memória da subárvore esquerda
        destroyRecursive(node->left);
        // Libera memória da subárvore direita
        destroyRecursive(node->right);
        // Libera o nó atual
        delete node;
    }
}

// Função main para demonstrar o uso da árvore
int main() {
    BinarySearchTree bst;
    
    // Inserindo valores
    cout << "Inserindo elementos na árvore: 50, 30, 70, 20, 40, 60, 80" << endl;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);
    bst.insert(60);
    bst.insert(80);
    
    // Imprimindo em ordem (deve estar ordenado)
    cout << "Elementos em ordem crescente: ";
    bst.printInOrder();
    
    // Buscando valores
    cout << "Buscando o valor 40: " << (bst.search(40) ? "Encontrado" : "Não encontrado") << endl;
    cout << "Buscando o valor 90: " << (bst.search(90) ? "Encontrado" : "Não encontrado") << endl;
    
    // Removendo valores
    cout << "Removendo o elemento 30..." << endl;
    bst.remove(30);
    
    cout << "Elementos após remoção: ";
    bst.printInOrder();
    
    return 0;
}