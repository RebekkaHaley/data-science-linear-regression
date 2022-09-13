import numpy as np

from data_science_linear_regression.linear_regression import MyLinearRegression

x_dummy = np.array([1, 2, 3])
y_dummy = np.array([10, 15, 20])

def test_my_linear_regression_default() -> None:
    my_regr = MyLinearRegression()
    assert my_regr.__dict__ == {}

def test_my_linear_regression_r() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.r == 1

def test_my_linear_regression_a() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.a == 5

def test_my_linear_regression_b() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.b == 5
