import numpy as np


class MyLinearRegression:
    """A linear regression model.

    Functions correspond to the calculation of Linear Regression by hand using the formula: y = a + bx.
    """
    def __init__(self):
        return


    def _calculate_r(self, x, y):
        """Calculates Pearson Correlation Coefficient (r).

        Args:
            x (np.array of float): list of independent variables
            y (np.array of float): list of dependent variables

        Returns:
            float: Pearson's correlation coefficient
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = sum([(x_i-x_mean)*(y[i]-y_mean) for i, x_i in enumerate(x)])
        sub_denominator_x = sum([(x_i-x_mean)**2 for i, x_i in enumerate(x)])
        sub_denominator_y = sum([(y_i-y_mean)**2 for i, y_i in enumerate(y)])
        denominator = np.sqrt(sub_denominator_x * sub_denominator_y)
        return numerator/denominator


    def _calculate_b(self, x, y, r):
        """Calculates slope (b) of regression line.

        Args:
            x (np.array of float): list of independent variables
            y (np.array of float): list of dependent variables
            r (float): Pearson's correlation coefficient
        
        Returns:
            float: slope value
        """
        x_stdev = np.std(x, ddof=1)
        y_stdev = np.std(y, ddof=1)
        return r * (y_stdev / x_stdev)


    def _calculate_a(self, x, y, b):
        """Calculates y-intercept (a) of regression line.

        Args:
            x (np.array of float): list of independent variables
            y (np.array of float): list of dependent variables
            b (float): slope value

        Returns:
            float: y-intercept value
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return y_mean - (b * x_mean)


    def fit(self, x, y):
        """Calculates coefficients for the formula: y = a + bx, and writes to self.

        Args:
            x (np.array of float): list of independent variables
            y (np.array of float): list of dependent variables

        Returns:
            None
        """
        self.r = self._calculate_r(x, y)
        self.b = self._calculate_b(x, y, r=self.r)
        self.a = self._calculate_a(x, y, b=self.b)
        return


    def predict(self, x):
        """Calculates predicted y using: y = a + bx.

        Args:
            x (np.array of float): list of independent variables

        Returns:
            np.array of float: list of predicted 'y' dependent variables
        """
        return np.array([self.a + (self.b * x_i) for x_i in x]).flatten()
