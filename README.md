## Welcome

This repo contains code that re-creates scikit-learn's *'linear model'* from scratch as a Python class.

The model made from scratch is available in a separate python script: *'linear_regression.py'*.

## Linear Regression

Linear regression is a statistical method used to determine the strength of relationship between a dependent variable *(usually 'y')* and an independent variable *(usually 'x')* based on a *'line of best fit'*.
It is also known as simple regression or ordinary least squares (OLS).

Linear regression can be depicted graphically as a straight line, where the slope *('b')* and y-intercept *('a')* are coefficients.

The model made from scratch uses the following formula to calculate the dependent variable, *'y'* : **y = a + bx**.

The slope *'b'* represents how change in one variable impacts a change in the other.

The y-intercept *'a'* represents the value of one variable when the value of the other is zero.
However, it can be unclear how to extract real-world meaning from the value of the y-intercept.
For example, if 'y' is household spending and 'x' is household income, what does it mean to have a positive (or negative) y-intercept when household income is zero?

It is important to note that linear regression itself does not indicate causation.
