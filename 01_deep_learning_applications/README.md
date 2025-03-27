# Big Themes Studied

## Chapter 2 - Introduction to Deep Neural Networks

1. Activation functions in Deep Learning
    - Activation functions are mathematical functions applied to the output of a neural network layer. They introduce non-linearity into the model, enabling it to learn complex patterns and relationships in the data. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh, each suited for different tasks and architectures.

2. Backpropagation in Deep Learning
    - Backpropagation is a supervised learning algorithm used for training neural networks. It calculates the gradient of the loss function with respect to the network's weights by propagating errors backward through the layers. This process enables the optimization algorithm to update the weights effectively, improving the model's accuracy.

3. Overfitting and Underfitting  
    - **Overfitting:** This occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization on unseen data. Overfitting can be mitigated using techniques like regularization, dropout, and increasing the training data.  
    - **Underfitting:** This happens when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test datasets. Addressing underfitting may involve increasing model complexity, training for more epochs, or improving feature engineering.

4. Loss function

    - The loss function measures the difference between the predicted output of the model and the actual target values. It serves as a guide for the optimization process, helping the model improve its predictions by minimizing this difference during training.

## Chapter 4 - Fundamentals of Neural Networks

1. Convolutional Neural Networks and Recurrent Neural Networks  
    - **Convolutional Neural Networks (CNNs):** These are specialized neural networks designed for processing structured grid data like images. They use convolutional layers to extract spatial features, making them highly effective for tasks like image recognition and object detection.  
    - **Recurrent Neural Networks (RNNs):** These are neural networks designed for sequential data, such as time series or text. They maintain a memory of previous inputs through recurrent connections, making them suitable for tasks like language modeling and speech recognition.

2. Optimization and Regularization  
    - **Optimization:** This refers to the process of adjusting the weights of a neural network to minimize the loss function. Common optimization algorithms include Gradient Descent, Adam, and RMSprop, which help improve the model's performance during training.  
    - **Regularization:** This involves techniques to prevent overfitting by penalizing complex models. Examples include L1/L2 regularization, Dropout, and Early Stopping, which ensure the model generalizes well to unseen data.  

3. Dropout and Batch Normalization  
    - **Dropout:** A regularization technique where randomly selected neurons are ignored during training. This prevents the network from becoming overly reliant on specific neurons, reducing overfitting.  
    - **Batch Normalization:** A method to normalize the inputs of each layer within a neural network. It stabilizes and accelerates training by reducing internal covariate shift, leading to improved performance.