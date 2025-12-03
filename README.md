The dataset used in this project can be downloaded from the link below:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


🧠 Brain Tumor Detection using CNN (Deep Learning)

This project uses Convolutional Neural Networks (CNNs) to classify MRI brain scan images into four tumor categories:

Glioma

Meningioma

Pituitary

No Tumor

The project is implemented in Python, using TensorFlow, Keras, NumPy,ImageNet, Matplotlib, and OpenCV.

📂 Folder Structure (IMPORTANT)

Users must follow this exact folder structure before running the notebook.

Brain_Tumor_Detection/
│
├── brain_tumer_cnn.ipynb
dataset/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/

│
└── README.md

📥 Dataset Download

Download the dataset from this link:

🔗 Dataset Link: <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

After downloading:

Create a folder named dataset

Inside it, create 4 subfolders for train and test:

glioma/

meningioma/

pituitary/

no_tumor/

Extract and place the images into their correct class folders.

⚙️ Installation
Install required libraries
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn

Place the dataset inside the dataset/ folder.

Open the file brain_tumer_cnn.ipynb in:

VS Code (with Python + Jupyter extensions)

Jupyter Notebook

Google Colab

Run all cells sequentially.

The model will:

Load images

Preprocess data

Train the CNN model

Evaluate accuracy

Predict tumor type

🧠 Model Architecture (CNN)

The model contains:

Conv2D layers

MaxPooling

Flatten

Dense layers

Dropout

Softmax output layer (4 classes)

