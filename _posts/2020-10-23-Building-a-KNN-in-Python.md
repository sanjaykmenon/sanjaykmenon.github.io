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
class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
    
    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(X_train)):
                dist = scipy.spatial.distance.euclidean(X_train[j] , X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)
    
    ```

We will pass the K while creating an object for the class ‘KNN’.

Fit method just takes in the training data, nothing else.

We used scipy.spatial.distance.euclidean for calculating the distance between two points.

Predict method runs a loop for every test data point, each time calculating distance between the test instance and every training instance. It stores distance and index of the training data together in a 2D list. It then sorts that list based on distance and then updates the list keeping only the K shortest distances(along with their indices) in the list.

It then pulls out labels corresponding to those K nearest data points and checks which label has the majority using Counter. That majority label becomes the label of the test data point.

Score method just compares our test output with their actual output to find the accuracy of our prediction.

