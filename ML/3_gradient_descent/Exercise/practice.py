import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('test_scores.csv')

x, y = df.math, df.cs

def gradient_descent(x,y):
    m_curr = b_curr = 0
    rate = 0.0001
    n = len(x)
    plt.scatter(x,y,color='red',marker='+',linewidth='5')
    prev_cost = [1000000,]
    for i in range(20):
        y_pred = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_pred)])
        if cost < min(prev_cost):
            prev_cost.append(cost)
    
#         print (m_curr,b_curr, i)
        plt.plot(x,y_pred,color='green')
        md = -(2/n)*sum(x*(y-y_pred))
        yd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * yd

        print("m: {}, b: {}, cost: {}, iteration: {}".format(m_curr, b_curr, cost, i))
        print(prev_cost)

gradient_descent(x, y)