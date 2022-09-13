import numpy as np
from sklearn import linear_model

from data_science_linear_regression.linear_regression import MyLinearRegression

x_dummy = np.array([1, 2, 3])
y_dummy = np.array([10, 15, 20])

def test_my_linear_regression_default() -> None:
    my_regr = MyLinearRegression()
    assert my_regr.__dict__ == {}

def test_my_linear_regression_fit_r() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.r == 1

def test_my_linear_regression_fit_a() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.a == 5

def test_my_linear_regression_fit_b() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    assert my_regr.b == 5

def test_my_linear_regression_predict_b() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    y_pred = my_regr.predict(x=x_dummy)
    assert y_pred[0] == 10
    assert y_pred[1] == 15
    assert y_pred[2] == 20

def test_my_linear_regression_equal_to_sklearn() -> None:
    my_regr = MyLinearRegression()
    my_regr.fit(x=x_dummy, y=y_dummy)
    my_y_pred = my_regr.predict(x=x_dummy)
    sk_regr = linear_model.LinearRegression()
    sk_regr.fit(x_dummy.reshape(-1, 1), y_dummy.reshape(-1, 1))
    sk_y_pred = sk_regr.predict(x_dummy.reshape(-1, 1)).flatten()
    np.testing.assert_allclose(my_y_pred, sk_y_pred, rtol=1e-5, atol=0)