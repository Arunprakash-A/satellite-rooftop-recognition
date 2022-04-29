# Rooftop recognition from Satellite Images

### Problem Statement

Given a coordinate (latitude-longitude) as an Input, design an AI model that
says whether that particular coordinate is on the rooftop or not. (Well, think about its applications )

### Approach

1. *Learning paradigm:* Supervised Learning as we can collect and label images from, say, Google's earth engine.
2. *Model:* Convolutional Neural Network (CNN) for binary classification.
3. *Rationale:* CNN's are capable of extracting complex features from
   images and, subsequently, they could be used to make a decision.

### Data Collection

#### Basic Details and challenges

1. *Source of Data:* Google Earth
2. *Input Format:* (Degree - arcmin - arcsecond), where 1o =108 Km,
   10 = 1.8 Km and 1" = 30 m.
3. However, the maximum Resolution provided by Google Earth is limited
   to 30 m (i.e, 1 arc second) along both latitude and the longitude. Therefore,
   the maximum resolution of a cell is 900 m^2   and the coordinate of the
   cell is considered for the labelling process.
4. *Challenge*: As a consequence, a single 900 m2 cell may contain more
   than one rooftop and/or a non-rooftop region.
5. *Labelling:* A cell is labelled as 1 (positive sample) if more than half
   of the region of a cell contains a rooftop. It is labelled as 0 (negative
   sample) otherwise.
6. Total number of samples: 201
7. Number of samples in the Training set: 157
8. Number of samples in Testing set: 44

### Sample Images

Each image is of 1 arcsec resolution.

### CNN Model

1. The ConvNet Architecture used in the project is shown in Fig.3
2. Activation: All neurons use Relu activation except the neuron in the
   output layer. Output neuron uses sigmoid activation function.
3. *Loss function:* Binary Cross Entropy.
4. *Optimization:*  Gradient Descent with momentum
5. *Number of learnable parameters*: 89957

### Performance

### Output Predictions

### Running the Code

**Required libraries** to execute the code: `Pytorch, sklearn, numpy, matplotlib`

1. Unzip the zipped folder named code and keep it in the current working
   directory.
2. All the training and testing images are stored in a directory named
   "data".
3. The trained weights and metrics are stored in a directory named "chk-
   points".
4. To test the model with pre-trained weights, enter `"python test.py"` in
   CLI.
5. To evaluate the model, type " python evaluation.py 10". The argument
   10 denotes the number of images to test randomly from the test dataset.
6. To plot the training metrics such as Loss and Accuracy, type `"python`
   `plot metrics.py"`

### # Happy Learning

