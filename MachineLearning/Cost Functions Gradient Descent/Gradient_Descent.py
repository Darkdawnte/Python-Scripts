import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import math

df = pd.read_excel('data.xlsx')

def sklearn_method(independent_variable, target):
    reg = LinearRegression()
    reg.fit(independent_variable,target)
    return reg.coef_ , reg.intercept_


def gradient_descent(x, y):
    m_current = b_current = 0
    iteration = 1000000
    n = len(x)
    learning_rate = 0.0002
    last_cost = 0
    
    for i in range(iteration):
        y_predicted = m_current * x + b_current
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * bd
        if math.isclose(cost, last_cost, rel_tol=1e-20):
            break
        last_cost = cost
        
        print("m: {}  b: {} cost: {} iteration: {}".format(m_current, b_current, cost, i))
    return m_current , b_current
        
maths = np.array(df.math)
cs = np.array(df.cs)
m , b = gradient_descent(maths, cs)
print('with gradient descent function coef:{} and intercept:{}'.format(m, b))
m_sklearn , b_sklearn = sklearn_method(df[['math']], cs)
print('with sklearn method coef:{} and intercept:{}'.format(m_sklearn, b_sklearn))