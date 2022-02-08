import math
from scipy import integrate
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def __main__():
    data_csv_file = 'data.csv'
    [np_views, np_subs] = parseData(data_csv_file)
    print(np_views)
    print(np_subs)
    daysToGraphForViews = generateXAxisArray(len(np_views))
    daysToGraphForSubs = generateXAxisArray(len(np_subs))

    figure, axes = plt.subplots(2)
    figure.suptitle("Growth of YouTube Views and Subscribers Over Days")
    regression = runLinearRegression(np_views)
    graphLinearRegression(axes[0],
                          daysToGraphForViews, np_views, regression.coef_, regression.intercept_, "Days", "Views")
    print(f"f(y) = {regression.coef_}x + {regression.intercept_}")

    regression = runLinearRegression(np_subs)
    graphLinearRegression(axes[1],
                          daysToGraphForSubs, np_subs, regression.coef_, regression.intercept_, "Subs", "Views")
    print(f"f(y) = {regression.coef_}x + {regression.intercept_}")

    plt.show()


def parseData(data_csv_file):
    delimited_file = pd.read_csv(
        data_csv_file, usecols=[0, 1, 2, 3])
    np_views = np.array(delimited_file['Views'])
    np_subs = np.array(delimited_file['Subscribers'])
    return [np_views, np_subs]


def runLinearRegression(dataToRegress):
    X = generateXAxisArray(len(dataToRegress))
    X = X.reshape(-1, 1)
    regression = LinearRegression().fit(X, dataToRegress)
    return regression


def graphLinearRegression(axes, xAxis, yAxis1, coefficient, intercept, xLabel, yLabel):
    yAxis2 = evaluateLinearRegression(coefficient,
                                      intercept,
                                      xAxis)

    axes.plot(xAxis, yAxis1)
    axes.plot(xAxis, yAxis2)
    axes.set_xlabel(xLabel)
    axes.set_ylabel(yLabel)
    axes.text(xAxis[len(xAxis) - 1], yAxis2[len(yAxis2) - 1],
              f"f(y) = {coefficient}x + {intercept}")
    axes.fill_between(xAxis, 0,
                      yAxis1, facecolor='blue', alpha=0.5)


def evaluateLinearRegression(coefficient, intercept, variable):
    return coefficient * variable + intercept


def runIntegrate(parseData):
    pass


def generateXAxisArray(len):
    return np.arange(len)


__main__()
