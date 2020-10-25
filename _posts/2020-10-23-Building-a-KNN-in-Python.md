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


