import numpy as np 
import matplotlib.pyplot as plt

def solve(X, Y):
  '''
    uses analytical method to find regression line
    slope = ⵉ(X - X') / ⵉ(Y - Y')
    intercep = Y' - slope * X'
  '''
  global ln

  # initialize accumulators
  numerator = denominator = 0
  xmean = np.mean(X)
  ymean = np.mean(Y)

  # calculate numerator and denominator of slope
  for i in range(X.shape[0]):
    numerator += (X[i] - xmean) * (Y[i] - ymean)
    denominator += (X[i] - xmean) ** 2

  slope = numerator/denominator
  intercept = ymean - slope*xmean

  draw_residual_plots(X, Y, intercept, slope)
  draw(slope, intercept)

def draw_residual_plots(X, Y, intercept, slope):
  '''
    plots residual plots for the data points X, Y 
    for regression line with hyperparameter theta
  '''
  global ln

  for i in range(X.shape[0]):
    if ln[i] != i:
      ln[i].remove()
    ln[i], = plt.plot(np.array([X[i], X[i]]), np.array([Y[i], slope*X[i]+intercept]),
                  color = "yellow", label = "residual plot")

def draw(slope, intercept):
  '''
    plots the regression line with given slope and intercept
  '''
  x = np.array([-100,100])
  line.set_xdata(x)
  line.set_ydata(slope * x + intercept)
  plt.title("Slope = %.2f Intercept = %.2f" % (slope, intercept))
  plt.draw()


def onclick(event):
  '''
    Adds new point to the data and 
    applies gradient descent on new dataset
  '''
  global X, Y, n

  # add the point to the data
  X = np.append(X, event.xdata)
  Y = np.append(Y, event.ydata)

  # plot the new point
  plt.scatter(X, Y, color = "blue", label = "data points")

  # if alteast 2 points, apply linear regression
  if X.shape[0] >= 2:
    solve(X, Y)


if __name__ == '__main__':

  # initialize dataset
  X = np.array([])
  Y = np.array([])

  # array to save residual plots
  # used to remove the revious plots
  ln = [x for x in range(100)]

  # bind the click event handler
  fig = plt.figure()
  fig.canvas.mpl_connect('button_press_event', onclick)

  # initialize graph 
  axes = plt.gca()
  plt.scatter(X, Y, color = "blue", label = "data points")
  plt.plot([], [], color = "yellow", label = "residual plot")
  line, = axes.plot(X, Y, color = "red", label = "regression line")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Linear Regression Live Plot")
  plt.legend()
  plt.xlim(-100,100)
  plt.ylim(-200,200)
  plt.show()
