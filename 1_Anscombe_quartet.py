# import libraries
import json
import numpy as np 
import matplotlib.pyplot as plt

# linear regression - returns slope and intercept
def solve(X, Y):
  
  n = len(X)
  
  # initialize accumulators
  numerator = denominator = 0
  xmean = np.mean(X)
  ymean = np.mean(Y)

  # calculate numerator and denominator of slope
  for i in range(n):
    numerator += (X[i] - xmean) * (Y[i] - ymean)
    denominator += (X[i] - xmean) ** 2

  slope = numerator/denominator
  intercept = ymean - slope*xmean

  return slope, intercept

# draw subplot with given data x and y in quadrant quad
def draw(quad, x, y):

  X = np.array([0,20])
  plt.subplot(2, 2, quad)
  plt.scatter(x, y)
  slope, intercept = solve(x, y)

  plt.title("Slope = %f \n Intercept = %f" % (slope, intercept))
  plt.plot(X, slope*X + intercept, "r-")

if __name__ == "__main__":
  # load data - anscombe's quartet
  with open("data/anscombe.json", "r") as file:
    data = json.load(file)

  # initialize graph
  plt.figure(figsize=(12,12))
  plt.xlim(0,20)
  plt.ylim(0,20)

  # draw anescomb's subplots
  draw(1, data["x"], data["y1"])
  draw(2, data["x"], data["y2"])
  draw(3, data["x"], data["y3"])
  draw(4, data["x4"], data["y4"])

  plt.show()