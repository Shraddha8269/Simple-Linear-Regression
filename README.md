#Import All libraries:
import numpy as np
from matplotlib import pyplot as plt

#defining gradient descent function
def gradient_decent(x,y,iterations):
    m_curr=b_curr=0
    l=0.008
    cost_=[]
    m=len(x)
    for i in range(0,iterations):
       y_pred=m_curr*x+b_curr        #Calculating the hypothesis function y=b0+b1x
       cost=1/m*sum([val**2 for val in (y-y_pred)])  #Calculating the cost function
       md=-2/m*sum(x*(y-y_pred))     #Updating the coefficients simultaneoulsy
       bd=-2/m*sum(y-y_pred)
       m_curr-=l*md
       b_curr=-l*bd
       cost_.append(cost)
    return(m_curr,b_curr,cost_)
    
    
#  Applying linear regression
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,12])
m_curr,b_curr,cost=gradient_decent(x,y,1000)
print("The values of parameters are:")
print("theta0",b_curr)
print("theta1",m_curr)

#Plotting the curve between cost function and number of iterations
plt.plot(np.arange(1000), cost, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Error vs Iterations')
plt.show()
