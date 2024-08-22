# Leaf-Type-Classification-System-with-ANN/CNN
This project is a Leaf Type Classification System developed in Python that uses Artificial Neural Networks (ANN) or Convolutional Neural Networks (CNN) to classify different types of leaves based on color features. The system leverages the power of libraries such as OpenCV, NumPy, scikit-learn, and TensorFlow to achieve accurate and efficient classification.

Project Overview
The main objective of this project is to classify different types of leaves using machine learning techniques. The project includes:

Data Preprocessing: Resizing and normalizing images, and converting labels into a binary format for the ANN model and categorical encoding for the CNN model.
Model Development:
For the ANN model, a fully connected network architecture with hidden layers is used.
For the CNN model, convolutional layers followed by pooling and dropout layers are implemented.
Training: The models are trained on a labeled dataset of leaf images with configurable hyperparameters like learning rate and the number of epochs.
Evaluation: The performance of the models is evaluated using accuracy and loss metrics, with visualizations to track model improvement over epochs.
GUI Interface: A simple GUI is created using PyQt5 to load and classify leaf images.

![image](https://github.com/user-attachments/assets/6bbab649-6491-44d2-8ef2-e211d16a6642)


Features
Image Classification: Classify leaf images into different categories such as Nangka, Daun Sirih, etc.
Model Flexibility: Switch between ANN and CNN models based on your requirements.
User-Friendly Interface: A GUI built with PyQt5 for easy interaction with the classification system.
Performance Visualization: Plots showing the training and validation accuracy and loss across epochs.
Real-Time Testing: Test the model with new images to see real-time classification results.
Technologies Used
Python: The primary programming language used for this project.
TensorFlow: Used for building and training the neural network models.
OpenCV: Utilized for image processing tasks.
NumPy: Employed for handling numerical operations on image data.
scikit-learn: Used for preprocessing and evaluating the model.
PyQt5: Implemented for creating the graphical user interface.
How to Run the Project
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/Leaf-Type-Classification-System-with-ANN-or-CNN.git
cd Leaf-Type-Classification-System-with-ANN-or-CNN
Install Dependencies:
Make sure you have Python installed. Then, install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the GUI:
Start the application by running the main file:

bash
Copy code
python ANN.py
Test the Model:
Use the GUI to load a leaf image and see the classification results.

Results
The system achieves high accuracy in classifying leaf types, with the CNN model outperforming the ANN model due to its ability to capture spatial hierarchies in images.

Future Improvements (COMING SOON)
Expand Dataset: Include more leaf types and a larger dataset for better generalization.
Model Optimization: Experiment with different architectures and hyperparameters to improve model performance.
Deploy as a Web App: Make the system accessible online through a web interface.
