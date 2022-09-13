from data_science_linear_regression.linear_regression import MyLinearRegression

def test_my_linear_regression_default() -> None:
    my_regr = MyLinearRegression()
    print(my_regr.__class__)