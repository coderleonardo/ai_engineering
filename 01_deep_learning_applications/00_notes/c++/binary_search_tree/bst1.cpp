// A Binary Search Tree (BST) is a hierarchical data structure where:
// 1. Each node contains a value.
// 2. The left child of a node contains values smaller than the node's value.
// 3. The right child of a node contains values greater than the node's value.
// 4. Both left and right subtrees must also be binary search trees.
//
// Key Properties:
// - Efficient for searching, insertion, and deletion with an average time complexity of O(log n) if balanced.
//
// Example:
// For the values {8, 3, 10, 1, 6}, the BST would look like this:
//       8
//      / \
//     3   10
//    / \
//   1   6
//
// This structure allows quick lookups by traversing left or right based on comparisons.

#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* left;
    Node* right;

    Node(int value) : data(value), left(nullptr), right(nullptr) {}
};

class BinarySearchTree {
public:
    Node* root;

    BinarySearchTree() : root(nullptr) {}

    void insert(int value) {
        root = insert_rec(root, value);
    }

    void inorder_traversal() {
        inorder_rec(root);
        cout << endl;
    }

private:
    Node* insert_rec(Node* node, int value) {
        if (node == nullptr) {
            return new Node(value);
        }
        if (value < node->data) {
            node->left = insert_rec(node->left, value);
        } else if (value > node->data) {
            node->right = insert_rec(node->right, value);
        }
        return node;
    }

    void inorder_rec(Node* node) {
        if (node != nullptr) {
            inorder_rec(node->left);
            cout << node->data << " ";
            inorder_rec(node->right);
        }
    }
};

int main() {
    BinarySearchTree bst;

    bst.insert(8);
    bst.insert(3);
    bst.insert(10);
    bst.insert(1);
    bst.insert(6);

    cout << "Inorder Traversal of BST: ";
    bst.inorder_traversal();

    return 0;
}
