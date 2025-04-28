# Big Themes Studied

## Chapter 2 - Introduction to Deep Neural Networks

### Activation functions in Deep Learning
- Activation functions are mathematical functions applied to the output of a neural network layer. They introduce non-linearity into the model, enabling it to learn complex patterns and relationships in the data. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh, each suited for different tasks and architectures.

### Backpropagation in Deep Learning
- Backpropagation is a supervised learning algorithm used for training neural networks. It calculates the gradient of the loss function with respect to the network's weights by propagating errors backward through the layers. This process enables the optimization algorithm to update the weights effectively, improving the model's accuracy.

### Overfitting and Underfitting  
- **Overfitting:** This occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization on unseen data. Overfitting can be mitigated using techniques like regularization, dropout, and increasing the training data.  
- **Underfitting:** This happens when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test datasets. Addressing underfitting may involve increasing model complexity, training for more epochs, or improving feature engineering.

### Loss function
- The loss function measures the difference between the predicted output of the model and the actual target values. It serves as a guide for the optimization process, helping the model improve its predictions by minimizing this difference during training.

## Chapter 4 - Fundamentals of Neural Networks

### Convolutional Neural Networks and Recurrent Neural Networks  
- **Convolutional Neural Networks (CNNs):** These are specialized neural networks designed for processing structured grid data like images. They use convolutional layers to extract spatial features, making them highly effective for tasks like image recognition and object detection.  
- **Recurrent Neural Networks (RNNs):** These are neural networks designed for sequential data, such as time series or text. They maintain a memory of previous inputs through recurrent connections, making them suitable for tasks like language modeling and speech recognition.

### Optimization and Regularization  
- **Optimization:** This refers to the process of adjusting the weights of a neural network to minimize the loss function. Common optimization algorithms include Gradient Descent, Adam, and RMSprop, which help improve the model's performance during training.  
- **Regularization:** This involves techniques to prevent overfitting by penalizing complex models. Examples include L1/L2 regularization, Dropout, and Early Stopping, which ensure the model generalizes well to unseen data.  

### Dropout and Batch Normalization  
- **Dropout:** A regularization technique where randomly selected neurons are ignored during training. This prevents the network from becoming overly reliant on specific neurons, reducing overfitting.  
- **Batch Normalization:** A method to normalize the inputs of each layer within a neural network. It stabilizes and accelerates training by reducing internal covariate shift, leading to improved performance.

## Chapter 6 - Understanding the Transformers - Part 0

### Optimization in Machine Learning  
- Optimization is the process of minimizing the loss function to improve the performance of a machine learning model. In neural networks, this involves adjusting the weights and biases through iterative updates. The process typically includes two key steps:  
    - **Feedforward:** The input data is passed through the network, and predictions are generated. The loss function then calculates the error between the predicted and actual values.  
    - **Backpropagation:** The error is propagated backward through the network to compute the gradients of the loss function with respect to the weights. These gradients guide the optimization algorithm in updating the weights to reduce the error.  

- A generic formula for the optimization process is:  
    $$w_{t+1} = w_t - \eta \cdot \nabla L(w_t)$$
    Where:  
    - $w_t$: Current weights at iteration $t$  
    - $\eta$: Learning rate (step size)  
    - $\nabla L(w_t)$: Gradient of the loss function with respect to the weights  
    - $w_{t+1}$: Updated weights for the next iteration  

- Common optimization algorithms include Gradient Descent, Stochastic Gradient Descent (SGD), and Adam, each with unique strategies for updating weights efficiently.

## Chapter 7 - Understanding the Transformers - Part 1

### Fully Connected Neural Networks  
- Fully Connected Neural Networks (FCNNs), also known as Dense Networks, are the simplest type of artificial neural networks where each neuron in one layer is connected to every neuron in the next layer. These networks are versatile and can be used for a variety of tasks, including classification and regression. However, they may not be as efficient as specialized architectures like CNNs or RNNs for certain types of data, such as images or sequences.

### TF-IDF Method  
- **TF-IDF (Term Frequency-Inverse Document Frequency):** This is a statistical method used in natural language processing to evaluate the importance of a word in a document relative to a collection of documents (corpus).  
    - **Term Frequency (TF):** Measures how frequently a term appears in a document.  
    - **Inverse Document Frequency (IDF):** Measures how unique or rare a term is across the corpus.  
- The TF-IDF score is calculated as:  
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$  
    Where:  
    - $t$: Term  
    - $d$: Document  
    - $\text{TF}(t, d)$: Term frequency of $t$ in $d$  
    - $\text{IDF}(t)$: Inverse document frequency of $t$  
- TF-IDF is widely used in text mining and information retrieval tasks, such as search engines and document classification.

### Activation Functions
- Activation functions play a critical role in neural networks by introducing non-linearity, which allows the model to learn and represent complex patterns in the data. Without non-linearity, the network would behave like a linear model, regardless of its depth, limiting its ability to solve complex problems.

    - **Neuron Activation:** Activation functions determine whether a neuron should be activated or not based on the weighted sum of its inputs. This decision impacts the flow of information through the network.

    - **Vanishing and Exploding Gradients:** During backpropagation, gradients can become very small (vanishing) or very large (exploding), especially in deep networks. This can hinder the training process. Activation functions like ReLU help mitigate the vanishing gradient problem by maintaining larger gradients for positive inputs, while careful weight initialization and gradient clipping can address exploding gradients.

### Weight Initialization Techniques

Weight initialization is a crucial step in training neural networks, as it impacts the convergence speed and stability of the optimization process. Proper initialization helps prevent issues like vanishing or exploding gradients.

1. **Glorot Initialization (Xavier Initialization):**  
    - Balances the variance of activations and gradients across layers by initializing weights with values drawn from a distribution with variance dependent on the number of input and output units. Suitable for activation functions like Sigmoid and Tanh.

2. **He Initialization:**  
    - Designed for ReLU and its variants, this method initializes weights with a variance proportional to the number of input units, helping to maintain stable gradients in deep networks.

3. **Orthogonal Initialization:**  
    - Initializes weights as orthogonal matrices, ensuring that the weight matrix preserves the magnitude of input vectors. This is particularly useful for RNNs to maintain long-term dependencies.

4. **Uniform Initialization:**  
    - Assigns weights randomly from a uniform distribution within a specified range. While simple, it may require careful tuning to avoid gradient issues.

5. **Batch Initialization:**  
    - Refers to initializing weights differently for each batch during training. This is less common but can be used in specific scenarios to introduce variability.

Proper weight initialization, combined with techniques like Batch Normalization, can significantly improve the training dynamics of neural networks.

### L2 Regularization

L2 regularization, also known as weight decay, is a technique used to prevent overfitting in deep neural networks by adding a penalty term to the loss function. This penalty is proportional to the sum of the squared values of the model's weights. The modified loss function can be expressed as:

$$L_{\text{total}} = L_{\text{original}} + \lambda \sum_{i} w_i^2$$

Where:  
- $L_{\text{total}}$: Total loss with regularization  
- $L_{\text{original}}$: Original loss function  
- $\lambda$: Regularization strength (hyperparameter)  
- $w_i$: Weight of the $i$-th parameter  

By discouraging large weight values, L2 regularization helps the model generalize better to unseen data, reducing the risk of overfitting.

### Learning Rate Scheduling

Learning rate scheduling is a technique used to adjust the learning rate during training to improve convergence and model performance. By modifying the learning rate over time, the model can escape poor local minima and converge more effectively. Common types of learning rate schedules include:

1. **Step Decay:**  
    - Reduces the learning rate by a fixed factor after a predefined number of epochs.  
    - Example: Halving the learning rate every 10 epochs.

2. **Exponential Decay:**  
    - Decreases the learning rate exponentially over time using a decay rate.  
    - Formula:  
      $$\eta_t = \eta_0 \cdot e^{-\lambda t}$$  
      Where:  
      - $\eta_t$: Learning rate at time $t$  
      - $\eta_0$: Initial learning rate  
      - $\lambda$: Decay rate  

3. **Cosine Annealing:**  
    - Gradually reduces the learning rate following a cosine curve, often resetting periodically.  
    - Useful for cyclical learning rate schedules.

4. **Cyclical Learning Rate (CLR):**  
    - Alternates the learning rate between a minimum and maximum value within a cycle.  
    - Helps the model explore different regions of the loss surface.

5. **Warm Restarts:**  
    - Periodically resets the learning rate to a higher value and then decays it, mimicking a restart in training.  
    - Often combined with cosine annealing.

Learning rate scheduling can be implemented using libraries like PyTorch or TensorFlow, and it is a powerful tool to enhance training efficiency and model performance.

### Early Stopping

Early stopping is a regularization technique used to prevent overfitting during training by monitoring the model's performance on a validation dataset. Training is halted when the validation performance stops improving for a specified number of epochs (patience). This ensures the model does not overfit the training data while maintaining good generalization to unseen data.

### Metrics for Binary Classification

Evaluating the performance of a binary classification model involves using various metrics that provide insights into its accuracy, precision, recall, and other aspects. Below are some of the most commonly used metrics:

#### 1. Accuracy
- **Definition:** Accuracy measures the proportion of correctly classified instances (both positive and negative) out of the total instances.
- **Formula:**  
    $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$  
    Where:  
    - TP: True Positives  
    - TN: True Negatives  
    - FP: False Positives  
    - FN: False Negatives  
- **Use Case:** Accuracy is a good metric when the dataset is balanced, but it may be misleading for imbalanced datasets.

#### 2. Precision
- **Definition:** Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive.
- **Formula:**  
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$  
- **Use Case:** Precision is important when the cost of false positives is high, such as in spam detection or medical diagnosis.

#### 3. Recall (Sensitivity or True Positive Rate)
- **Definition:** Recall measures the proportion of actual positive instances that are correctly identified by the model.
- **Formula:**  
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$  
- **Use Case:** Recall is crucial when the cost of false negatives is high, such as in disease detection or fraud detection.

#### 4. F1-Score
- **Definition:** The F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both.
- **Formula:**  
    $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$  
- **Use Case:** F1-Score is useful when there is an uneven class distribution and a balance between precision and recall is desired.

#### 5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Definition:** The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate (FPR) at various threshold levels. The AUC represents the area under this curve.
- **Formula for FPR:**  
    $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$  
- **Use Case:** AUC-ROC is a threshold-independent metric that evaluates the model's ability to distinguish between classes.

#### 6. KS Statistic (Kolmogorov-Smirnov)
- **Definition:** KS measures the maximum difference between the cumulative distribution of true positives and false positives. It indicates the model's ability to separate the positive and negative classes.
- **Use Case:** KS is widely used in credit risk modeling and other domains where class separation is critical.

#### 7. Lift
- **Definition:** Lift measures the improvement of the model's predictions over random guessing. It is often used in marketing and customer targeting.
- **Formula:**  
    $$\text{Lift} = \frac{\text{Precision in a specific decile}}{\text{Overall Precision}}$$  
- **Use Case:** Lift is useful for evaluating the effectiveness of a model in identifying high-value segments.

#### 8. Log Loss (Logarithmic Loss)
- **Definition:** Log Loss evaluates the uncertainty of the model's predictions by penalizing incorrect predictions with higher confidence.
- **Formula:**  
    $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$  
    Where:  
    - $y_i$: Actual label (0 or 1)  
    - $p_i$: Predicted probability for class 1  
- **Use Case:** Log Loss is commonly used in probabilistic models to assess prediction confidence.

#### 9. Matthews Correlation Coefficient (MCC)
- **Definition:** MCC is a balanced metric that considers all four confusion matrix categories (TP, TN, FP, FN). It is especially useful for imbalanced datasets.
- **Formula:**  
    $$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$  
- **Use Case:** MCC provides a comprehensive evaluation of model performance.

#### 10. Specificity (True Negative Rate)
- **Definition:** Specificity measures the proportion of actual negative instances that are correctly identified by the model.
- **Formula:**  
    $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$  
- **Use Case:** Specificity is important when the cost of false positives is high.

### Choosing the Right Metric
The choice of metric depends on the problem domain, class distribution, and the cost of false positives and false negatives. For example:
- Use **Precision** and **Recall** for imbalanced datasets.
- Use **Accuracy** for balanced datasets.
- Use **ROC-AUC** and **KS** for evaluating ranking models.
- Use **Lift** for marketing and customer segmentation.

## Chapter 8 - Understanding the Transformers - Part 2

### LSTM Architecture

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) designed to address the vanishing gradient problem and capture long-term dependencies in sequential data. LSTMs achieve this through a unique architecture that includes memory cells and gating mechanisms:

1. **Memory Cell:** Stores information over time, allowing the network to retain important features from earlier time steps.

2. **Input Gate:** Controls how much new information is added to the memory cell.

3. **Forget Gate:** Determines how much information from the memory cell should be discarded.

4. **Output Gate:** Regulates the information passed to the next layer or time step.

These gates use sigmoid and tanh activation functions to manage the flow of information, enabling LSTMs to effectively model sequences with long-term dependencies, such as time series, text, and speech data.

### Leaky ReLU

Leaky ReLU (Rectified Linear Unit) is a variant of the ReLU activation function that addresses the "dying ReLU" problem, where neurons can become inactive and stop learning if their outputs are always zero. Unlike ReLU, which outputs zero for negative inputs, Leaky ReLU allows a small, non-zero gradient for negative inputs. The function is defined as:

$$f(x) = 
\begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}$$

Where $\alpha$ is a small positive constant (e.g., 0.01). This ensures that the network can still learn from negative inputs, improving training stability and performance in some cases.

### Word Embeddings

Word embeddings are a type of word representation that captures the semantic meaning of words in a continuous vector space. Unlike traditional one-hot encoding, which represents words as sparse and high-dimensional vectors, word embeddings map words to dense, low-dimensional vectors, enabling models to understand the relationships between words more effectively.

#### Key Concepts

1. **Semantic Similarity:**  
    - Words with similar meanings are mapped to vectors that are close to each other in the embedding space. For example, the vectors for "king" and "queen" would be closer than those for "king" and "car."

2. **Dimensionality Reduction:**  
    - Word embeddings reduce the dimensionality of word representations while preserving meaningful relationships, making them computationally efficient for downstream tasks.

3. **Contextual Relationships:**  
    - Embeddings capture the context in which words appear, allowing models to understand polysemy (words with multiple meanings) and synonyms.

#### Popular Word Embedding Techniques

1. **Word2Vec:**  
    - Developed by Google, Word2Vec uses two architectures:  
      - **Continuous Bag of Words (CBOW):** Predicts a target word based on its surrounding context words.  
      - **Skip-Gram:** Predicts the context words given a target word.  
    - Word2Vec learns embeddings by maximizing the likelihood of word co-occurrence in a given context.

2. **GloVe (Global Vectors for Word Representation):**  
    - Developed by Stanford, GloVe combines local context (word co-occurrence) and global statistics (word frequency) to generate embeddings. It uses a matrix factorization approach to capture word relationships.

3. **FastText:**  
    - Developed by Facebook, FastText extends Word2Vec by representing words as a combination of character n-grams. This allows it to handle out-of-vocabulary words and capture subword information.

4. **ELMo (Embeddings from Language Models):**  
    - Developed by AllenNLP, ELMo generates contextualized word embeddings by considering the entire sentence. It uses a bidirectional LSTM to capture word meaning based on context.

5. **BERT (Bidirectional Encoder Representations from Transformers):**  
    - Developed by Google, BERT produces contextual embeddings using a transformer-based architecture. It considers both left and right contexts simultaneously, making it highly effective for tasks like question answering and sentiment analysis.

#### Applications of Word Embeddings

1. **Natural Language Processing (NLP):**  
    - Tasks like sentiment analysis, machine translation, and text classification benefit from the semantic understanding provided by embeddings.

2. **Information Retrieval:**  
    - Word embeddings improve search engines by enabling semantic matching between queries and documents.

3. **Recommendation Systems:**  
    - Embeddings are used to model user preferences and item similarities in recommendation systems.

4. **Question Answering and Chatbots:**  
    - Embeddings help models understand user queries and generate contextually relevant responses.

#### Advantages of Word Embeddings

- **Semantic Understanding:** Captures relationships between words, enabling better generalization.  
- **Efficiency:** Reduces the dimensionality of word representations, improving computational performance.  
- **Transfer Learning:** Pre-trained embeddings can be fine-tuned for specific tasks, saving time and resources.

#### Challenges

- **Out-of-Vocabulary Words:** Traditional embeddings like Word2Vec struggle with unseen words, though models like FastText address this issue.  
- **Context Independence:** Static embeddings (e.g., Word2Vec, GloVe) do not account for word meaning changes based on context, which is resolved by contextual embeddings like BERT and ELMo.

## Chapter 9 - Deep Learning for Computer Vision Task### 
