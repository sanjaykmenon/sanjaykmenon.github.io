---
title: Building a kNN implementation in Python without libraries
---

## What is kNN?

kNN stands for k-Nearest Neighbors. It is a supervised machine learning algorithm used to solve classification problems. The kNN algorithm assumes that similar objects exist in close proximity in a k dimensional space. The image below shows how similar data points are close to each other. 


![kNN Image](/assets/knn.png "kNN Image")

The algorithm works under the idea that similar objects are at an 'X' distance from each other and we can find other similar objects by measuring the distance between them. In our case, we will be measuring the Euclidean distance between various objects to find similar points.

## Steps of the kNN Algorithm

- Initialize K to the number of nearest neighbors

- Calculate the Euclidean distance between the query example and the current example from the data

- Add the distane and index of the example to an ordered list

- Sort the ordered list of distances from smallest to largest in ascending order

- Pick the first K entries from the sorted list and return the mode

I’ve made a split, for training our algorithm and then testing it on the test data separately.

We will define a class ‘KNN’ inside which we will define every essential function that will make our algorithm work. We will be having the following methods inside our class.

fit: As discussed earlier, it’ll just keep the data with itself, since KNN does not perform any explicit training process.

Distance: We will calculate Euclidean distance here.

Predict: This is the phase where we will predict the class for our testing instance using the complete training data. We will implement the 3 stepped process discusses above in this method.
    
Score: Finally We’ll have a score method, to calculate the score for our model based on the test data

What about ‘K’ ?: The most important guy here is K, we will pass ‘K’ as an argument while initializing the object for our KNN class(inside __init__)

Here is a good picture I got for an idea about various distance metrics used:
Each one of them can be easily calculated using their Numpy inbuild functions or can also be coded directly if you want.

![Distance Image](/assets/distance.png "Distance Formulae")

Here is the code snipped for the class

```
# Imports
import numpy as np


# Class
class k_nearest_neighbors:
    """Class of the K Nearest Neighbors Algorithm implementation"""

    """
    **Implementation**
    Method:
    - euclidean_distance(a, b):
        Returns euclidian distance of values
        
    - fit_knn(X_train, y_train):
        Fits model to training data

    - predict_knn(X):
        Returns predictions for X based on fitted model

    - display_knn(x)
        Returns list of nearest_neighbors + corresponding euclidian distance
    """

    # Initialization
    def __init__(self, n_neighbors=5):  # default neighbors to be returned
        """Init for algorithm"""
        self.n_neighbors = n_neighbors

    # Euclidian Distance
    def euclidean_distance(self, a, b):
        """
        Returns euclidian distance of values between row a and row b
        Inputs: a : int or float
                b : int or float
        Output: euclidian_distance : float
        """
        eucl_distance = 0.0  # initializing eucl_distance at 0

        for index in range(len(a)):
            """
            Based on: https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,metric%20as%20the%20Pythagorean%20metric.
            Calculation: Subtract b from a, square difference,
            add to eucl_distance.
            """
            eucl_distance += (a[index] - b[index]) ** 2

            euclidian_distance = np.sqrt(eucl_distance)

        return euclidian_distance

    # Fit k Nearest Neighbors
    def fit_knn(self, X_train, y_train):
        """
        Fits model to training data
        Inputs: X_train : array of int or float
                y_train : list or array of target
        Output: N/A pass along to predict_knn
        NOTE: I specifically choose not to include a "build-in"
        data split function, as it allows more flexibility in choosing
        data split methods according preference and/or data problem.
        """
        self.X_train = X_train
        self.y_train = y_train

    # Predict X for kNN
    def predict_knn(self, X):
        """
        Returns predictions for X based on fitted X_train and y_train data
        Inputs: X : list or array
        Output: prediction_knn : list of floats for each vector in X
        """

        # initialize prediction_knn as empty list
        prediction_knn = []

        # # initialize euclidian_distances as empty list
        # euclidian_distances = []

        for index in range(len(X)):  # Main loop iterating through len(X)

            # initialize euclidian_distances as empty list
            euclidian_distances = []

            for row in self.X_train:
                # for every row in X_train, find eucl_distance to X using
                # euclidean_distance() and append to euclidian_distances list
                eucl_distance = self.euclidean_distance(row, X[index])
                euclidian_distances.append(eucl_distance)

            # sort euclidian_distances in ascending order, and retain only k
            # neighbors as specified in n_neighbors (n_neighbors = k)
            neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

            # initialize dict to count class occurrences in y_train
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

            # max count labels to prediction_knn
            prediction_knn.append(max(count_neighbors, key=count_neighbors.get))

        return prediction_knn

    # Print/display list of nearest_neighbors + corresponding euclidian
    # distance
    def display_knn(self, x):
        """
        Inputs: x : vector x
        Output: display_knn_values : returns a list containing nearest
        neighbors and their correscponding euclidian distances
        to the vector x wrapped in tuples
        """

        # initialize euclidian_distances as empty list
        euclidian_distances = []

        # for every row in X_train, find eucl_distance to x
        # using euclidean_distance() and append to euclidian_distances list
        for row in self.X_train:
            eucl_distance = self.euclidean_distance(row, x)
            euclidian_distances.append(eucl_distance)

        # sort euclidian_distances in ascending order, and retain only k
        # neighbors as specified in n_neighbors (n_neighbors = k)
        neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

        # initiate empty display_knn_values list
        display_knn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_distances = euclidian_distances[index]
            display_knn_values.append(
                (neighbor_index, e_distances)
            )  # changed to list of tuples
        # print(display_knn_values)
        return display_knn_values
    
```

We will pass the K while creating an object for the class ‘KNN’.

Fit method just takes in the training data, nothing else.

We used scipy.spatial.distance.euclidean for calculating the distance between two points.

Predict method runs a loop for every test data point, each time calculating distance between the test instance and every training instance. It stores distance and index of the training data together in a 2D list. It then sorts that list based on distance and then updates the list keeping only the K shortest distances(along with their indices) in the list.

It then pulls out labels corresponding to those K nearest data points and checks which label has the majority using Counter. That majority label becomes the label of the test data point.

Score method just compares our test output with their actual output to find the accuracy of our prediction.

