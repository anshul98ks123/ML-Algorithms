import numpy as np
import matplotlib.pyplot as plt


def hypothesis(X, theta):
    '''
      calculates mx + c for each point in X
      where, m and c are theta[1] and theta[1]
    '''
    return theta[0] + theta[1] * X


def gradient(X, Y, theta, learning_rate):
    '''
      calculate change in theta according to the rule
      del theta[0] = sum [ learning_rate * (h(X) - Y) ]
      del theta[1] = sum [ learning_rate * (h(X) - Y) * X ]
    '''
    error = np.array([[0.0], [0.0]])
    temp = (learning_rate * (hypothesis(X, theta) - Y))

    error[0] = sum(temp)
    error[1] = np.matmul(temp, X.T)

    return error


def error(X, Y, theta):
    '''
      calculates mean squared error for given dataset X, Y
    '''
    return sum((hypothesis(X, theta) - Y) ** 2) / 2


def gradient_descent(X, Y, learning_rate=0.5):
    '''
      applies gradient descent on given dataset X, Y
    '''
    theta = np.array([[0.0], [0.0]])

    for _ in range(2000):
        theta = theta - gradient(X, Y, theta, learning_rate)/X.shape[0]

    draw_residual_plots(X, Y, theta)
    draw(theta[0], theta[1])


def draw_residual_plots(X, Y, theta):
    '''
      plots residual plots for the data points X, Y 
      for regression line with hyperparameter theta
    '''
    global ln

    for i in range(X.shape[0]):
        if ln[i] != i:
            ln[i].remove()
        ln[i], = plt.plot(np.array([X[i], X[i]]), np.array([Y[i], hypothesis(X[i], theta)]),
                          color="yellow", label="residual plot")


def onclick(event):
    '''
      Adds new point to the data and 
      applies gradient descent on new dataset
    '''
    global X, Y

    X = np.append(X, event.xdata)
    Y = np.append(Y, event.ydata)

    # plot the new point
    plt.scatter(X, Y, color="blue", label="data points")

    # if alteast 2 points, apply linear regression
    if X.shape[0] >= 2:
        gradient_descent(X, Y)


def draw(intercept, slope):
    '''
      plots the regression line with given slope and intercept
    '''
    global line
    x = np.array([-1, 1])
    line.set_xdata(x)
    line.set_ydata(slope * x + intercept)

    plt.title("Slope = %.2f Intercept = %.2f" % (slope, intercept))
    plt.draw()


if __name__ == '__main__':

    # initialize the dataset
    X = np.array([])
    Y = np.array([])

    ln = [x for x in range(100)]

    # bind the click event handler
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)

    # initialize graph
    axes = plt.gca()
    plt.scatter(X, Y, color="blue", label="data points")
    plt.plot([], [], color="yellow", label="residual plot")
    line, = axes.plot(X, Y, color="red", label="regression line")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Live Plot")
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
