# Active Learning for Quality Engineering
This repo contains the code for replicating the results from the paper on Active Learning published in Quality Engineering.

The images used in this paper are from the MVTec datasets and they can be requested [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

# **Active Learning for Industrial Applications**  

This repository contains the code for replicating results from the paper *Active Learning for Industrial Applications*, published in *Quality Engineering*. The project explores **active learning strategies** to improve data efficiency in real-world industrial applications, reducing labeling costs while maintaining high model performance.

## 📌 **Overview**
Supervised learning models often require **large labeled datasets**, which can be costly and time-consuming to obtain. **Active learning** addresses this by selecting the most informative samples for annotation, improving model efficiency with fewer labeled examples. This project implements and compares:

- **Random Sampling** – Baseline method selecting samples randomly.
- **Margin-Based Active Learning** – Selects uncertain samples near the decision boundary.
- **Entropy-Based Sampling** – Prioritizes samples with high uncertainty.
- **Hybrid Approaches** – Combines multiple active learning strategies to improve performance.

We apply these methods to **industrial quality control** and **fault detection** tasks, demonstrating how active learning improves labeling efficiency in manufacturing settings.

## 🛠 **Setup Instructions**
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/active-learning-industrial.git
   cd active-learning-industrial
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the example notebook:
   ```sh
   jupyter notebook active_learning.ipynb
   ```

## 📂 **Code Structure**

- `requirements.txt`: file that contains a list of all the packages installed in the environment used, along with their version numbers.
- `feature_extraction.ipynb`: notebook showing how to obtain preprocessed features from the images using a pre-trained ResNet-18.
- `sampling_strategies.py`: functions for performing random sampling and margin sampling.
-  `active_learning.ipynb`: notebook showing how the implementation of the active learning strategy.

## 📊 **Visualizing Active Learning**
### **Margin-Based Sampling**
Below is an illustration of margin-based sampling in action:

<img src='margin.png' width='500'/>

The red points indicate samples with the lowest confidence, which are selected for annotation in active learning.

### **Comparison of Active Learning Strategies**
From the paper, we analyze different active learning strategies in an industrial setting:

<img src='images/active_learning_comparison.png' width='600'/>

- **Baseline Model Performance** – Supervised learning without active selection.
- **Active Learning Performance** – Demonstrates improvement with fewer labeled samples.
- **Efficiency Gains** – Reduction in annotation costs while maintaining accuracy.

## 📜 **Citation**
If you use this code, please cite:
> Cacciarelli, D., & Kulahci, M. (2025). *Active Learning for Industrial Applications.* Quality Engineering.
