# VoteEnsemble

This repository contains the python code for the paper ''Subsampled Ensemble Can Improve Generalization Tail Exponentially'' (https://arxiv.org/pdf/2405.14741).


## Installation
1.&nbsp;cd to the root directory, i.e., VoteEnsemble,  of this repository.

2.&nbsp;Install the required dependencies, *setuptools*, *numpy*, *zstandard*, if not already intalled. You can install them directly with
```
pip install setuptools numpy zstandard
```
or
```
pip install -r requirements.txt
```

3.&nbsp;Install VoteEnsemble via
```
pip install .
```
## Quick Start
To use VoteEnsemble, you first need to define a base learning algorithm by subclassing the BaseLearner class. Below are two simple use cases to illustrate this.
### Linear regression
Consider a linear regression
<!-- $$
\min_{\theta} E[(Y - X^T\theta)^2]
$$ -->

![Equation](./images/LR.png)

where $X$ is the input vector, $Y$ is the response variable, and $\theta$ is the model parameter vector. The script exampleLR.py implements such an example, where the method of least squares is the base learning algorithm, and applies $\mathsf{ROVE}$ and $\mathsf{ROVEs}$ to learn the model parameters. Try the example by running (*scikit-learn* required)
```
python exampleLR.py
```
which shall produce the result
```
ROVE outputs the parameters:  [1.98771233e-15 1.00000000e+00 2.00000000e+00 3.00000000e+00
 4.00000000e+00 5.00000000e+00 6.00000000e+00 7.00000000e+00
 8.00000000e+00 9.00000000e+00]
ROVEs outputs the parameters:  [-4.43618871e-16  1.00000000e+00  2.00000000e+00  3.00000000e+00
  4.00000000e+00  5.00000000e+00  6.00000000e+00  7.00000000e+00
  8.00000000e+00  9.00000000e+00]
```
### Stochastic linear program
Consider a linear program with stochastic coefficients in the form of
<!-- $$
\begin{align*}
\min_{\theta}\  &E[z^T\theta]\\
\text{s.t.}\ &A\theta \leq b\\
&l \leq \theta \leq u
\end{align*}
$$ -->

![Equation](./images/LP.png)

where $z$ is the random coefficient vector, $A$ is the constraint matrix, $b$ is the right hand side, and $l,u$ are lower and upper bounds of the decison $\theta$. The script exampleLP.py implements such an example, where the sample average approximation is the base learning algorithm, and applies $\mathsf{MoVE}$, $\mathsf{ROVE}$, and $\mathsf{ROVEs}$ to obtain solution estimates. You can try the example by running (*cvxpy* required)
```
python exampleLP.py
```
which shall produce the result
```
MoVE outputs the solution:  [7.96814478e-10 4.65702752e-10 2.79592672e-10 8.89130869e+00
 7.66984396e+00 3.91509297e-10 5.58240876e+00 1.68686836e-09
 1.34440530e+00 9.62175569e-10]
ROVE outputs the solution:  [ 1.57771765e-09  1.95436889e-08 -1.74352733e-10  8.89130873e+00
  7.66984386e+00 -2.06028856e-10  5.58240860e+00  1.91247807e-07
  1.34440518e+00  1.66841730e-09]
ROVEs outputs the solution:  [5.96767908e-10 2.78273914e-09 4.01128399e-10 8.89130869e+00
 7.66984395e+00 2.37835199e-10 5.58240876e+00 3.49369934e-09
 1.34440530e+00 1.47623460e-09]
```
