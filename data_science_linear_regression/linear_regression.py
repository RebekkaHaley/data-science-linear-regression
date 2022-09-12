import numpy as np


class MyLinearRegression:
    """A linear regression model.

    Functions correspond to the calculation of Linear Regression by hand using the formula: y = a + bx.
    """
    def __init__(self):
        return


    def _calculate_r(self, x, y):
        """Calculates Pearson Correlation Coefficient (r).
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

        r: Pearson's correlation coefficient
        """
        x_stdev = np.std(x, ddof=1)
        y_stdev = np.std(y, ddof=1)
        return r * (y_stdev / x_stdev)


    def _calculate_a(self, x, y, b):
        """Calculates y-intercept (a) of regression line.

        b: Slope
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        return y_mean - (b * x_mean)


    def fit(self, x, y):
        """Calculates components of the linear regression formula: y = a + bx.
        """
        self.r = self._calculate_r(x, y)
        self.b = self._calculate_b(x, y, r=self.r)
        self.a = self._calculate_a(x, y, b=self.b)
        return


    def predict(self, x):
        """Calculates predicted y using: y = a + bx.
        """
        return np.array([self.a + (self.b * x_i) for x_i in x]).flatten()
