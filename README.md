The aim of this project is the create a ML module that can control the car in forza4 to drive correctly according to the navigation lines.

## How

1.**Collect dataset from player**: including screen recording and key input action.

2.**Preprocess**: cropping, color extracting, gray-scaling, warping, resizing  by opencv.

3.**Training**: use the image as input and keyboard action as output.

4.**Predicting**: recording the actual game screen as input and use the module to predict the keyboard action.

## Reference

- ##### [EthanNCai/AI-Plays-ForzaHorizon4](https://github.com/EthanNCai/AI-Plays-ForzaHorizon4)

  *The clarity of the lane lines has a great impact on the image. Some lane line is not clear. In addition, since the canny edge detection of OpenCV is used, there will be a lot of irrelevant data in the image, and the trained model is difficult to converge.(probably result from my tiny dataset)*

- ##### [uppala75/CarND-Advanced-Lane-Lines](https://github.com/uppala75/CarND-Advanced-Lane-Lines?tab=readme-ov-file)

  *using the warping approach from the repo. In the front view, the further away the lane line is, the smaller the pixels we get，it's hard to detect and lock on to future lane lines if their pixels and their footprint essentially get smaller and smaller。using bird-eyes view allow us for future planning.*

- ##### [jimhoggey/SelfdrivingcarForza](https://github.com/jimhoggey/SelfdrivingcarForza)


## Getting Started

#### Step1: Collecting the dataset

run `DataColleciton.py` to collect the image from the screen and the keyboard action. we do some preprocessing in this step by opencv, so that the dataset won't take too much space.

![image-20241224122237564](assets/image-20241224122237564.png)

#### Step2: Preprocessing

Run `ConvLSTM-DataPreprocess.py` for data preprocessing.

1. Convert to sequential data:

   Convert the input data and output labels into sequential data to fit the LSTM model’s input requirements. For each input sequence, extract the past *pass_frame* frames as input and the corresponding output labels as targets.

2. Balance the data:

   - Count the occurrences of each label.
   - Calculate the average number of occurrences for each label to set the balancing target.
   - For labels with fewer occurrences than the target, keep all data; for labels with more occurrences than the target, randomly sample the data to achieve balance.
   - Shuffle the balanced data.

3. One-hot encoding for labels:

   Use the *onehot_encode* function to apply one-hot encoding to the keyboard events.

#### Step3:  Model Training

Run `ConvLSTM-TrainModel.py` to load the preprocessed data and train the model.

The network layers used in this project include:

1. **ConvLSTM2D layer**: Used to process spatiotemporal data, capturing correlations across both time and space.
2. **BatchNormalization layer**: Standardizes the input for each layer to accelerate training and improve model stability.
3. **TimeDistributed layer**: Applies the MaxPooling2D layer to each time step of the input sequence.
4. **MaxPooling2D layer**: Downsamples the spatial dimensions of the input to reduce computation and memory usage.
5. **Flatten layer**: Flattens the multidimensional input into a one-dimensional array for the fully connected layers.
6. **Dense layer**: Applies a linear transformation and activation function to the input.
7. **Dropout layer**: Randomly drops a portion of the input units during training to prevent overfitting.

<img src="assets/convlstm_model_structure_plot.png" alt="convlstm_model_structure_plot" style="zoom:50%;" />

#### Step4: Model Prediction

With the game running，run `ConvLSTM-PlayModel.py`，which reads image from the screen, preprocesses it, and uses the trained model to predict the keyboard actions. The model’s predictions are then applied through *pyautogui* to control the car, keeping it on track with the navigation lines.



