# 3Signet Image Classification Task

## Overview

The **3Signet Image Classification Task** is an interactive web application that classifies images into one of 100 fine-grained classes from the CIFAR-100 dataset. The model leverages a pre-trained ResNet-50 architecture optimized with TensorFlow/Keras for efficient and accurate classification. Users can upload images, and the app will provide real-time predictions with a confidence score.

This project is built with the aim of showcasing the capabilities of deep learning in image classification and is designed to be accessible and user-friendly for anyone looking to explore machine learning in action.

---

## Features

- **Interactive Web Interface**:  
  The application is built using [Streamlit](https://streamlit.io/), providing an intuitive and easy-to-use interface for users. All you need to do is upload an image, and the app will classify it for you. The result is displayed in real-time along with a confidence score.

- **CIFAR-100 Dataset**:  
  The model predicts one of 100 fine-grained classes from the **CIFAR-100** dataset, which includes a wide variety of object categories like animals (e.g., dog, cat, frog) and vehicles (e.g., car, truck, airplane). The CIFAR-100 dataset is known for its small-sized images (32x32 pixels), which pose challenges for the model to correctly identify objects in various conditions.

- **Pre-trained Neural Network**:  
  The model is based on the **ResNet-50** architecture, a deep convolutional neural network known for its ability to handle complex image classification tasks. The ResNet model has been pre-trained on a large-scale dataset and fine-tuned for the CIFAR-100 dataset. After experimentation with various models, ResNet-50 was selected due to its balanced performance and efficient use of resources.  

  - **Model Performance**:  
    The model’s **validation accuracy** is between **41-43%**, which indicates that while the model performs reasonably well, there is still room for improvement. The classification accuracy can be improved by experimenting with different models, adjusting hyperparameters, or fine-tuning the dataset.

- **Real-time Predictions**:  
  The application provides **real-time predictions**. Once an image is uploaded, the model processes the image and classifies it within seconds. The predicted class is displayed along with a confidence score, providing users with an understanding of how certain the model is about its prediction.  

  [**Live Demo**](https://3signet-image-classification-task-fujey3wwxhk8rvkqqq264m.streamlit.app/)  

---

## How It Works

1. **Upload an Image**:  
   The user can upload an image (in `.jpg`, `.jpeg`, or `.png` format) via the Streamlit interface.

2. **Image Preprocessing**:  
   The uploaded image is resized to the size required by the model (32x32 pixels) and normalized, ensuring that the pixel values lie between 0 and 1. This step is crucial to match the input format that the ResNet-50 model expects.

3. **Prediction**:  
   After preprocessing, the image is passed to the ResNet-50 model. The model processes the image and outputs a set of class predictions with associated confidence scores. The class with the highest confidence is selected as the final prediction.

4. **Results Display**:  
   The predicted class and the confidence score are displayed to the user in real-time. For instance, if the uploaded image is classified as a “cat,” the app will display:  
   - **Predicted Class: Cat**  
   - **Confidence: 75%**  

---

## Installation and Setup

To run the application locally, follow these steps:

### Requirements
- Python 3.7 or later
- TensorFlow 2.x
- Streamlit
- PIL (Pillow)
- NumPy

### Steps to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/3Signet-Image-Classification-Task.git
    cd 3Signet-Image-Classification-Task
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
---

## Model and Evaluation

### Model Architecture
- **ResNet-50**:  
  The model uses the ResNet-50 architecture, a deep CNN that utilizes residual blocks to improve training by allowing gradients to flow more easily through the network. This architecture helps mitigate the vanishing gradient problem and allows the model to train deeper networks effectively.

### Performance Metrics
- **Validation Accuracy**: 41% - 43%
    - The model achieves moderate performance on the CIFAR-100 dataset. Given the complexity of classifying small images across 100 categories, these results are a solid starting point.
    - There is room for improvement by either adjusting the model architecture (e.g., using more layers or different models like DenseNet or EfficientNet) or enhancing the data preprocessing.

- **Next Steps for Improvement**:
    - Experiment with different learning rates, data augmentations, and optimizers to improve the model's generalization.
    - Consider using transfer learning with models trained on larger datasets (e.g., ImageNet).

---

## Challenges and Future Work

While the app is functional, several challenges remain:
1. **Accuracy Improvement**: The current model accuracy can be improved by exploring more advanced models or techniques like **transfer learning** or **ensemble models**.
2. **Handling Complex Images**: The model might struggle with images outside the CIFAR-100 dataset. Additional training on more varied datasets could help in improving predictions for diverse or noisy images.
3. **User Interface**: The Streamlit interface could be enhanced by allowing users to upload multiple images, compare predictions, or download results.

---

## Contributions

Feel free to contribute to this project! Suggestions, improvements, or bug fixes are always welcome. You can:
- Open an issue for any bugs or feature requests.
- Fork the repo and submit a pull request with your changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CIFAR-100 Dataset**: [Link to CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar-100.html)  
- **TensorFlow/Keras**: For their amazing deep learning framework and pre-trained models.  
- **Streamlit**: For providing an easy way to create interactive web apps.  
```
