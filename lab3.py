import numpy as np 
import matplotlib.pyplot as plt
from lab4 import *

no_points = 20
max_point = 100
min_point = -100


def f(x, a0, a1):
    return a0 + (a1*x)

x = np.random.uniform(low=min_point,high=max_point, size=no_points).tolist()
y = np.random.uniform(low=min_point,high=max_point, size=no_points).tolist()

'''
print(x)
print(y)
'''

'''
equations:
    Σyi = a0 N + a1 Σxi
    Σyi xi = a0 Σxi + a1 Σxi^2
'''
yi = 0
xi = 0
yixi = 0
xi2 = 0
complixity = 3
    
for i in range(no_points): #calculating all the sums
    yi += y[i]
    xi += x[i]
    yixi += y[i]*x[i]
    xi2 += x[i]**2

A = np.array([[no_points, xi],
              [xi, xi2]])
B = np.array([yi, yixi])

sol = np.linalg.solve(A,B)

a0 = sol[0]
a1 = sol[1]

x_plot=[]
y1_plot=[]
y2_plot = []

alpha = 0.2 
iterations = 10000

w,b = gradiant_decent(x,y,alpha,iterations)

for i in range(min_point, max_point):
    y1_plot.append(f(i,a0,a1))
    y2_plot.append(f(i,b,w))
    x_plot.append(i)



plt.plot(x_plot,y1_plot, 
         label = "line 1",
         color = 'green'
         )

plt.plot(x_plot,y2_plot, 
         label = "line 2",
         color = 'black'
         )
plt.legend()
plt.scatter(x,y,)
plt.show()
