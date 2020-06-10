import numpy as np
import matplotlib.pyplot as plt


def onclick(event):
    '''
        Adds new point to the data depending on click
        left click - class 1
        right click - class 2
    '''
    global X1, X2, Y1, Y2

    if event.button == 1:
        X1 = np.append(X1, event.xdata)
        Y1 = np.append(Y1, event.ydata)
        plt.scatter(X1, Y1, color="blue", label="class 1")

    else:
        X2 = np.append(X2, event.xdata)
        Y2 = np.append(Y2, event.ydata)
        plt.scatter(X2, Y2, color="green", label="class 2")

    if X1.shape[0] and X2.shape[0]:
        SVM(X1, Y1, X2, Y2)

    plt.draw()


def dist(X1, Y1, X2, Y2):
    ''' calculates euclidean distance '''
    return np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2)


def isLinearSeperable(slope, intercept, X1, Y1, X2, Y2, Print=False):
    ''' checks whether hyperplane seperates both classes '''

    d1 = np.array([Y1 - (slope * X1 + intercept)])
    d2 = np.array([Y2 - (slope * X2 + intercept)])

    return ((d1 < 0).all() and (d2 > 0).all()) or ((d1 > 0).all() and (d2 < 0).all())


def SVM(X1, Y1, X2, Y2):
    '''
        finds decision hyperplane by finding perpendicular
        bisector of line joining the support vectors
    '''
    mx1 = mx2 = my1 = my2 = -100
    mindist = 100000

    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            temp = dist(X1[i], Y1[i], X2[j], Y2[j])

            slope = (X2[j] - X1[i]) / (Y1[i] - Y2[j])
            intercept = ((Y1[i] + Y2[j]) / 2) - slope * ((X1[i] + X2[j]) / 2)

            # if pair of points linearly seperates and has less distance
            # than current support vectors, then make them new support vector
            if temp < mindist and isLinearSeperable(slope, intercept, X1, Y1, X2, Y2):
                mx1, my1, mx2, my2 = X1[i], Y1[i], X2[j], Y2[j]
                mindist = temp

    try:
        # slope and intercept of perpendicular bisector of Support Vectors
        slope = (mx2 - mx1) / (my1 - my2)
        intercept = ((my1 + my2) / 2) - slope * ((mx1 + mx2) / 2)

        draw_hyperplane(intercept, slope, (mx1 + mx2) / 2)

    except:
        plt.title(
            "No seperating hyperplane perpendicular to line joining support vectors\n")


def draw_hyperplane(intercept, slope, sv_x):
    '''
      plots the regression line with given slope and intercept
    '''
    global line

    if slope == np.inf:
        line.set_xdata([sv_x, sv_x])
        line.set_ydata([-1, 1])
    else:
        line.set_xdata([-1, 1])
        line.set_ydata(slope * np.array([-1, 1]) + intercept)

    plt.title("Slope = %.2f Intercept = %.2f" % (slope, intercept))


if __name__ == '__main__':

    # initialize the dataset
    X1 = np.array([])
    X2 = np.array([])

    Y1 = np.array([])
    Y2 = np.array([])

    # bind the click event handler
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)

    # initialize graph
    axes = plt.gca()
    plt.scatter(X1, Y1, color="blue", label="class 1")
    plt.scatter(X2, Y2, color="green", label="class 2")

    line, = axes.plot([], [], color="red", label="hyperplane")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SVM Live Plot")
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
