# Test Plan - Part II

When an algorithm is tested on a problem, it is run on the problem and the best cost, route, and runtime are recorded. Results are saved in MLflow along with the hyperparameters used for each algorithm. Success is measured by the algorithm with the lowest overall cost. If two or more algorithms have the same cost, they will be ranked by their runtime.

## Initial Pass of Testing on Unique Problem Shapes

Unique problem shapes and types have been selected. These include

- Asymmetric instances where d[i,j] != d[j,i]
- Cluster instances where 2+ cities form 2+ clusters in Euclidean space
- Spiral instances where the dispersion of cities represents a spiral
- Star instances where the dispersion of cities represents something of an asterisk or traditional hand-drawn star
- Tetrahedron instances where the dispersion of cities represents a 2D tetrahedron

Problems in each of these categories will be run against the project algorithms (ACO and Q-Learning). Results will be analyzed to determine if the problem shape or structure has any effect on results, ex. is any algorithm particularly good with any unique problem shape, or does the best performing algorithm, MMAS, generally perform best on all problem shapes?

### Points of Interest

* Q-Learning performing better than ACO for larger problems
* NNS performs the best on clusters
* Spiral is easy for all algorithms to handle

## Deep Dive based on Initial Pass

If results from the Initial Pass test are promising, a deep dive into any promising problem types can be undergone. For example, if star-shape problems are solved particularly well by any 1 algorithm, then more examples will be generated to see if this was an anomaly or if the algorithm actually consistently performs better on a particular problem type.

## Hyperparameter Tuning

Hyperparameters should be adjusted to their ideal values. This can be done with Optuna.

### Points of Interest

* ACO not much variability with hyperparameter tuning
* Q-Learning much variability with hyperparameter tuning, sometimes as good as ACO, ex. with brazil 58, got as good as 27145 when both ACO algorithms returned 27384. Seemed to do better when more myopic.

## Algorithm Improvement

If no other test routes are promising, there is room for improvement on the Q-Learning Algorithm. While it is currently the worse performing algorithm, it is possible that using it in conjunction with deep learning, i.e. a neural network, would show promising results. This is an area of interest that could prove promising if no other testing is fruitful, or ends in a timely manner.