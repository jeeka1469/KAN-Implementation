# KAN-Implementation
This is an implementation of Kolmogorov–Arnold Networks

This repository contains the implementation of Kolmogorov–Arnold Networks (KAN) for image classification and comparison with a traditional MLP (Multilayer Perceptron). The project demonstrates how KAN models can be used for training deep learning models and compares their performance with MLP models.

## Project Overview

- **KAN**: A deep learning model that uses kernel-based activation functions for better feature extraction.
- **MLP**: A traditional feedforward neural network model for comparison.
- The project also includes Streamlit for a simple UI to interact with the models.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/jeeka1469/KAN-Implementation.git
    cd KAN-Implementation
    ```

2. Create a virtual environment (recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. If you don't have the `requirements.txt` file yet, you can generate it with:

    ```bash
    pip freeze > requirements.txt
    ```

## Usage

### Running the Models

To test the models, you can either run the Streamlit app or directly evaluate the models through Python scripts.

- **Streamlit UI**: To launch the Streamlit app and interact with the models:

    ```bash
    streamlit run main.py
    ```

    This will start a local server, and you can interact with the KAN and MLP models via your browser.

- **Command Line**: You can also evaluate the models directly by running:

    ```bash
    python evaluate.py
    ```

### Models

- **MLP Model**: A simple Multilayer Perceptron model that classifies MNIST images.
- **KAN Model**: A Kernel-based Activation Network that classifies MNIST images with a custom activation function based on kernels.

## Model Performance

- **MLP Accuracy**: ~95.39%
- **KAN Accuracy**: ~97.03%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- https://arxiv.org/pdf/2404.19756
