# SORT (Simple Online and Realtime Tracking)

SORT is an algorithm used for tracking multiple objects in video sequences. It is designed to be simple, efficient, and capable of running in real-time. SORT uses a combination of the Kalman filter and the Hungarian algorithm to predict the positions of objects and associate detections with existing tracks.

## Key Components
- **Kalman Filter**: Used for predicting the future position of objects based on their previous states.
- **Hungarian Algorithm**: Used for solving the assignment problem, i.e., matching detected objects with predicted tracks.

## Example
Consider a video sequence where we need to track multiple cars. SORT will:
1. Detect the cars in each frame.
2. Predict the next position of each car using the Kalman filter.
3. Match the detected cars with the predicted positions using the Hungarian algorithm.
4. Update the tracks with the new detections.

# Deep SORT

Deep SORT extends the SORT algorithm by incorporating appearance information to improve tracking performance, especially in scenarios with occlusions and similar-looking objects. It uses a deep learning-based feature extractor to generate appearance descriptors for detected objects.

## Key Components
- **Appearance Descriptor**: A deep learning model that extracts features from detected objects to create unique identifiers.
- **Kalman Filter**: Used for predicting the future position of objects.
- **Hungarian Algorithm**: Used for matching detections with predicted tracks.

## Example
In the same video sequence of tracking multiple cars, Deep SORT will:
1. Detect the cars in each frame.
2. Extract appearance features for each detected car using a deep learning model.
3. Predict the next position of each car using the Kalman filter.
4. Match the detected cars with the predicted positions using both the Hungarian algorithm and the appearance descriptors.
5. Update the tracks with the new detections and appearance information.

By incorporating appearance information, Deep SORT can handle more complex tracking scenarios and maintain accurate tracks even when objects look similar or are temporarily occluded.