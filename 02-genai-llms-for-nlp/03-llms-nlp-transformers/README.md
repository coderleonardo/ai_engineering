# LLMs, Transformers and NLP

## Initial Transformer Architectures: BERT, GPT-3, and GPT-4

The Transformer architecture revolutionized Natural Language Processing by introducing the **self-attention mechanism**, allowing models to weigh the importance of different words in a sequence when processing it. This led to significant breakthroughs in various NLP tasks. Here's a look at some pivotal initial architectures:

---

### BERT (Bidirectional Encoder Representations from Transformers)

* **Architecture:** BERT utilizes a **bidirectional encoder** from the Transformer architecture. This means it considers the context from both the left and the right of each word in a sentence simultaneously during training.
* **Pre-training Tasks:** BERT was pre-trained on two main tasks:
    * **Masked Language Modeling (MLM):** Randomly masking some tokens in the input and the model's objective is to predict the original masked word based on the context of the other unmasked words.
    * **Next Sentence Prediction (NSP):** Training the model to understand the relationship between pairs of sentences, predicting whether the second sentence is the next sentence in the original document. (Note: The effectiveness of NSP has been debated in subsequent research).
* **Key Features:** Bidirectional encoding for rich contextual representations.
* **Use Cases:** BERT excels in tasks that require understanding the context of a given text:
    * **Text Classification:** Sentiment analysis, topic classification, spam detection.
    * **Named Entity Recognition (NER):** Identifying and classifying entities.
    * **Question Answering:** Extracting answers from a given text passage.
    * **Natural Language Inference (NLI):** Determining the logical relationship between two sentences (e.g., entailment, contradiction, neutral).
    * **Token Classification:** Tasks like part-of-speech tagging.

---

### GPT-3 (Generative Pre-trained Transformer 3)

* **Architecture:** GPT-3 employs a **decoder-only** Transformer architecture. It focuses on predicting the next token in a sequence, conditioned on the preceding tokens. It is significantly larger than its predecessors, with billions of parameters.
* **Pre-training Task:** Primarily trained on **causal language modeling (CLM)**, where the model predicts the next word in a sequence.
* **Key Features:** Massive scale, strong few-shot learning capabilities (performing tasks with only a few examples).
* **Use Cases:** GPT-3 demonstrates impressive abilities in generative tasks:
    * **Text Generation:** Writing articles, stories, poems, scripts, emails, and more.
    * **Translation:** Translating between languages.
    * **Code Generation:** Generating code in various programming languages.
    * **Question Answering:** Answering questions in a free-form manner.
    * **Summarization:** Generating summaries of longer texts.
    * **Creative Content Generation:** Creating dialogues, musical pieces, etc.

---

### GPT-4

* **Architecture:** GPT-4 is the successor to GPT-3, also believed to be a decoder-only Transformer architecture but with even greater scale and capabilities. The exact architectural details are largely proprietary.
* **Pre-training Task:** Similar to GPT-3, focused on next-token prediction on a vast and diverse dataset.
* **Key Features:** Improved reasoning abilities, better handling of complex instructions, enhanced coherence and fluency in generated text, and multimodal capabilities (though initially text-focused, it's designed to handle different modalities).
* **Use Cases:** GPT-4 expands upon GPT-3's capabilities with enhanced performance:
    * **Advanced Content Creation:** Generating more sophisticated and nuanced text.
    * **Complex Reasoning and Problem Solving:** Assisting with tasks requiring deeper understanding and inference.
    * **Improved Conversational AI:** Powering more natural and contextually aware chatbots.
    * **Creative Applications:** Generating more intricate and imaginative content.
    * **Multimodal Applications (evolving):** Integrating text with other modalities like images.

---

### Newer LLM Model Types (Further Study)

Here are some newer types and prominent examples of LLMs that you can explore further:

* **Mixture of Experts (MoE) Models:** Architectures that activate only a subset of their parameters for each input, leading to increased capacity with potentially lower computational cost per token (e.g., **Switch Transformers**, some versions of **GPT-4**, **Mixtral 8x7B**).
* **Decoder-Encoder Models for Generative Tasks:** While initially used for translation (e.g., **T5**, **BART**), these architectures are being adapted for more general generative tasks.
* **Efficient Transformer Variants:** Models focusing on reducing the computational cost and memory footprint of the self-attention mechanism, allowing for longer context windows and more efficient training and inference (e.g., **Longformer**, **Performer**, **FlashAttention**).
* **Instruction-Tuned Models:** Models specifically fine-tuned on datasets of instructions and desired outputs, making them better at following natural language instructions (e.g., **InstructGPT**, **FLAN T5**, **Llama 2-Chat**).
* **Open-Source LLMs:** Increasingly powerful and accessible models released by research labs and the open-source community, fostering innovation and accessibility (e.g., **Llama 2**, **Mistral 7B**, **Falcon**, **BLOOM**).
* **Multimodal LLMs:** Models capable of processing and generating content across multiple modalities, such as text, images, audio, and video (e.g., **GPT-4 with Vision**, **Gemini**, **Flamingo**, **BLIP-2**).
* **Retrieval-Augmented Generation (RAG) Models:** Models that combine the generative power of LLMs with the ability to retrieve relevant information from external knowledge sources to improve factual accuracy and reduce hallucinations (e.g., using models like **GPT-3.5 Turbo** or **Llama 2** in a RAG pipeline).

---

## LLM Evaluation Metrics

Evaluating Large Language Models (LLMs) is crucial to understand their performance across various tasks. Unlike traditional classification models, evaluating generative models often involves comparing generated text to reference text, which can be complex due to the inherent creativity and variability of language. Here are some common metrics:

---

### 1. BLEU (Bilingual Evaluation Understudy)

* **Purpose:** Primarily used for **machine translation** and other text generation tasks where fluency and adequacy are important. It measures the "n-gram precision" of the generated text against one or more reference texts.
* **How it Works:** It counts how many n-grams (sequences of N words) in the candidate text appear in the reference text. Higher overlap means a higher score. It also penalizes short generated sentences.
* **Range:** 0 to 1 (or 0 to 100), where higher is better.
* **Example:**
    * **Reference:** "The cat sat on the mat."
    * **Candidate 1:** "The cat sat on mat." (High BLEU)
    * **Candidate 2:** "The the the cat." (Low BLEU)
    * **Candidate 3:** "Sitting on the mat was the cat." (Lower BLEU than 1 due to less exact n-gram overlap, even if semantically similar)
* **Limitations:**
    * Doesn't directly measure semantic meaning or grammatical correctness.
    * Can give low scores to perfectly good sentences that use different phrasing.
    * Relies heavily on exact word matches, missing synonyms or paraphrases.

---

### 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

* **Purpose:** Widely used for **summarization** and other text generation tasks where recall (capturing information from the reference) is critical. It measures the overlap of n-grams, word sequences, or skip-bigrams between the generated text and a set of reference summaries.
* **Types:**
    * **ROUGE-N:** Measures n-gram overlap (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).
    * **ROUGE-L:** Measures the Longest Common Subsequence (LCS) to capture sentence-level structure.
    * **ROUGE-S:** Measures skip-bigram overlap (allowing gaps).
* **Range:** 0 to 1, where higher is better.
* **Example (for ROUGE-1 F1-score):**
    * **Reference:** "Police killed the gunman."
    * **Candidate:** "Police kill gunman."
    * Here, "police", "kill", "gunman" are common words. ROUGE-1 would capture the unigram overlap.
* **Limitations:** Similar to BLEU, it's primarily a lexical overlap metric and doesn't fully capture semantic meaning or conciseness for summarization.

---

### 3. METEOR (Metric for Evaluation of Translation with Explicit Ordering)

* **Purpose:** An improvement over BLEU for **machine translation**, aiming to address some of BLEU's limitations. It considers not just exact word matches but also synonyms, stems, and paraphrases.
* **How it Works:** Calculates scores based on aligning unigrams between candidate and reference sentences, considering exact, stem, synonym, and paraphrase matches. It also includes a penalty for fragmentation.
* **Range:** 0 to 1, where higher is better.
* **Example:** If "car" is in the reference and "automobile" is in the candidate, METEOR might give partial credit due to synonymy, whereas BLEU might not.
* **Strengths:** Often correlates better with human judgment than BLEU.

---

### 4. Perplexity

* **Purpose:** Primarily used to evaluate **language models** (how well a model predicts a sample of text). It measures how well the probability distribution learned by the model predicts the next item in a sequence.
* **How it Works:** Lower perplexity indicates the model is more confident and accurate in predicting the next token. It's the exponential of the average negative log-likelihood of the test data.
* **Example:** If a model assigns high probabilities to the actual next words in a test set, its perplexity will be low. If it assigns low probabilities (meaning it's "surprised" by the actual next words), perplexity will be high.
* **Limitations:** While good for intrinsic evaluation of language models, it doesn't directly measure the quality of generated text for specific tasks (e.g., summarization quality).

---

### 5. Token Loss / Cross-Entropy Loss

* **Purpose:** This is the primary **loss function used during training** of most LLMs (especially decoder-only models like GPT). It quantifies how "wrong" the model's predictions are compared to the actual next token.
* **How it Works:** For each predicted token, it calculates the negative logarithm of the probability assigned by the model to the *correct* next token. The goal during training is to minimize this loss.
* **Range:** Positive values, where lower is better.
* **Example:** If the true next word is "cat" and the model predicts "dog" with 90% probability and "cat" with 1% probability, the loss for "cat" will be very high. If it predicts "cat" with 90% probability, the loss will be very low.
* **Relationship to Perplexity:** Perplexity is derived directly from the cross-entropy loss.

---

### 6. Word Error Rate (WER)

* **Purpose:** Predominantly used for **speech recognition** and, less commonly, for evaluating the output of language generation systems in specific contexts (e.g., automatic transcription). It measures the number of errors (substitutions, deletions, insertions) required to transform the generated text into the reference text.
* **How it Works:** Calculated using dynamic programming to find the optimal alignment between the generated and reference sequences.
* **Range:** Percentage, where lower is better (0% means perfect match).
* **Example:**
    * **Reference:** "The quick brown fox"
    * **Candidate:** "A quick new fox"
    * Errors: "A" for "The" (substitution), "new" (insertion), "brown" (deletion). WER would reflect these.
* **Limitations:** Like BLEU, it's very sensitive to exact word matches and doesn't account for semantic similarity.

---

### Other Important Aspects & Emerging Metrics

While the above are common quantitative metrics, they often fall short in fully capturing the nuance of LLM outputs. Therefore, human evaluation is frequently indispensable. Additionally, more specialized and advanced metrics are emerging:

* **Human Evaluation:** Gold standard for many tasks. Involves human annotators rating output quality based on criteria like fluency, coherence, relevance, factual accuracy, and helpfulness.
* **Factuality Metrics:** Specific measures or benchmarks (e.g., using knowledge graphs or external search) to check if generated information is factually correct, especially for RAG systems.
* **Coherence/Consistency Scores:** Automated metrics attempting to capture how logical and well-structured a generated long-form text is.
* **Toxicity/Bias Scores:** Metrics to detect and quantify harmful content or biases in generated text.
* **Task-Specific Metrics:** For specific applications, custom metrics might be developed (e.g., for code generation, checking if the code compiles or passes test cases).
* **Adversarial Evaluation:** Testing models with inputs designed to expose vulnerabilities or biases.

---
