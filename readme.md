# Chess Piece Classification using CNN

## Description

This project implements a Convolutional Neural Network (CNN) to classify chess pieces. It uses TensorFlow and Keras to build and train a model that can identify five different chess pieces: king, queen, rook, bishop, and pawn.

## Features

- Image classification of chess pieces
- Utilizes Convolutional Neural Networks (CNN)
- Data augmentation for improved model performance
- Easy-to-use Google Colab implementation

## Requirements

- TensorFlow
- Keras
- scikit-learn
- Google Colab (for easy execution)

## How to Run

1. Download the Jupyter notebook file:
   [klasifikasi_catur.ipynb](https://github.com/rezapace/Machine-Learning-chess-classification/blob/master/klasifikasi_catur.ipynb)

2. Open [Google Colab](https://colab.research.google.com/?authuser=0#create=true)

3. Upload the downloaded `.ipynb` file to Google Colab

4. Run the following commands in a Colab cell to clone the repository and set up the environment:

   ```python
   !apt install git
   !git clone https://github.com/rezapace/Machine-Learning-chess-classification
   ```

5. Execute the cells in order:
   - Import required libraries
   - Prepare the dataset
   - Set up data augmentation and loading
   - Build the CNN model
   - Compile the model
   - Train the model
   - Evaluate the model
   - Save the model (optional)

6. The final cell will display the model's accuracy on the validation set.

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- 3 MaxPooling layers
- Flatten layer
- Dense layer with ReLU activation
- Dropout layer (0.5) for regularization
- Output layer with Softmax activation for 5 classes

## Results

The model achieves an accuracy of approximately 42.40% on the validation set after 10 epochs of training.

## Future Improvements

- Increase the dataset size
- Experiment with different model architectures
- Implement transfer learning using pre-trained models
- Fine-tune hyperparameters for better performance

## Contributing

Contributions to improve the model or extend the project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

For more information or to report issues, please visit the [project repository](https://github.com/rezapace/Machine-Learning-chess-classification).