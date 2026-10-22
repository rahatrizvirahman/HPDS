# What carries over from Assignment 1

We keep the supplied data and KNN decision rule. For each test instance, compute Euclidean distances to training instances, keep the closest k entries, and predict the most frequent class among them. The class column is a label, not a feature in the distance calculation.

The serial implementation scans training rows in their original order and inserts a new candidate only if its distance is strictly smaller. An equal-distance row therefore follows the earlier row. Class votes are examined in ascending class order, so a tied vote selects the smallest class number.

For a query at zero and training rows at minus one and plus one, both distances equal one. If the first row has class 1 and the second has class 0, k equal to one selects class 1. With k equal to two, both rows vote and class 0 wins the class tie. The synthetic test in `tests/check_predictions.py` exercises these two different tie rules.

Every test query is independent. We can assign it to a separate CUDA thread without changing the order in which that thread examines training rows. This adapts the thread indexing in Lecture 8, especially pages 9–16 and `vectorAdd.cu`, to the same query decomposition used in Assignment 1.

The copied historical report, measurements, and scripts live in `Reference_assignment1/`. The implementation folder now generates fresh Assignment 2 measurements. The datasets and `libarff` remain unchanged.

The serial algorithm remains the reference. Its surrounding code now validates inputs, writes optional predictions, frees its flattened matrices, and reports fractional milliseconds. Accuracy is counted directly rather than using a confusion matrix whose dimensions depend on the test labels. This allows a test set to omit a class that still appears among predictions.

Matching overall accuracy does not prove that two classifiers made the same predictions. One might correct one row and misclassify another while keeping the same score. We therefore compare every predicted class before trusting a GPU measurement.
