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
To use VoteEnsemble, you need to define a base learning algorithm by subclassing the BaseLearner class. Below are two simple use cases to illustrate this.
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
True model parameters = [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
ROVE outputs the parameters: [3.92431628e-03 1.01457683e+00 1.96402875e+00 3.01047031e+00
 4.00479241e+00 5.00741279e+00 5.99596621e+00 7.01602010e+00
 7.99180409e+00 8.99286178e+00]
ROVEs outputs the parameters: [-7.11339923e-03  1.00764019e+00  1.97278415e+00  3.00220791e+00
  3.99707439e+00  5.02509414e+00  5.98887793e+00  7.02417495e+00
  8.01337643e+00  8.96901555e+00]
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
True optimal objective value = -15.349275441094298
MoVE outputs the solution: [1.35655906e-09 9.94023566e+00 7.95126945e-10 2.57448639e+00
 5.77780176e-08 1.25721531e-09 1.83272283e-08 1.46573859e-09
 6.97747247e+00 5.22424867e-08], objective value = -15.294764680702862
ROVE outputs the solution: [2.31131990e-10 1.13393702e-10 1.22780890e-10 9.48606929e+00
 7.14193413e+00 3.15064932e-10 4.10652774e+00 1.76364405e+00
 7.82081479e-10 2.73901352e-10], objective value = -15.174436790910121
ROVEs outputs the solution: [7.31309987e-10 9.94023571e+00 3.63672223e-10 2.57448641e+00
 1.00839088e-08 3.12572614e-10 1.24294896e-08 4.55585503e-09
 6.97747247e+00 1.54772390e-09], objective value = -15.294764698873003
```
