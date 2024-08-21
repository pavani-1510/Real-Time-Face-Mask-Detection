# Real-Time Face Mask Detection

This project implements a real-time face mask detection system using computer vision techniques. The system can identify whether a person is wearing a face mask or not, and it can be applied in various scenarios such as public safety monitoring, automated access control, and health protocols.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Kaggle Notebook](#kaggle-notebook)

## Introduction

The Real-Time Face Mask Detection system is designed to monitor and enforce health safety protocols by automatically detecting the presence of face masks on individuals in real-time. It utilizes computer vision methods to process video streams or images and determine whether a person is wearing a mask.

## Features

- Real-time face mask detection.
- Capable of processing both video streams and static images.
- High accuracy under various lighting and background conditions.
- Integration capability with security systems for automated access control.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/pavani-1510/Real-Time-Face-Mask-Detection.git
    cd Real-Time-Face-Mask-Detection
    ```

2. **Install required dependencies:**
    Make sure you have Python installed. Then, install the necessary libraries by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application:**
    To start the face mask detection system, execute:
    ```bash
    python Model_Building_and_Evaluation.py
    ```

## Usage

The system can be used in real-time by connecting a webcam or providing video files. It will detect faces in the video and classify them as either "With Mask" or "Without Mask." The output will be displayed in a window with the classification label and a bounding box around the detected face.

## Dependencies

This project uses the following libraries:

- OpenCV
- TensorFlow or Keras (if used)
- Numpy
- Imutils

Ensure these dependencies are installed in your Python environment.

## Kaggle Notebook

Alternatively, you can run the Kaggle notebook for this project. Here is the link to the notebook: [Real-Time Face Mask Detection using ML](https://www.kaggle.com/code/rpavani2005/real-time-face-mask-detection-using-ml)
