import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def __main__():
    """Main entry point of the program"""
    data_csv_file = 'data.csv'
    [np_views, np_subs] = parseData(data_csv_file)
    viewsRegression = runLinearRegression(np_views)
    subsRegression = runLinearRegression(np_subs)
    chartGrowthOfSubsAndViews(
        np_views, viewsRegression, np_subs, subsRegression)

    daysToProject = daysLeftInYear()
    currentViews = 398000
    currentSubs = 5459
    projectGrowthOfSubsAndViews(
        viewsRegression,  subsRegression, daysToProject, currentViews, currentSubs)


def daysLeftInYear():
    """Returns the number of days left in the year"""
    return 365 - int(pd.Timestamp.now().strftime("%j"))


def projectGrowthOfSubsAndViews(viewsRegression, subsRegression, daysToProject, currentViews, currentSubs):
    """Projects the growth of subs and views for the days specified."""
    daysToGraphForViews = generateXAxisArray(daysToProject)
    daysToGraphForSubs = generateXAxisArray(daysToProject)

    projectedViews = evaluateLinearRegression(
        viewsRegression.coef_, viewsRegression.intercept_, daysToGraphForViews)
    projectedSubs = evaluateLinearRegression(
        subsRegression.coef_, subsRegression.intercept_, daysToGraphForSubs)

    cumulativeViews = predictSubs(
        daysToGraphForViews, projectedViews, currentViews, "Views")
    cumulativeSubs = predictSubs(
        daysToGraphForSubs, projectedSubs, currentSubs, "Subs")

    print(
        f"End of year prediction is currently: {math.floor(cumulativeSubs)} subscribers and {math.floor(cumulativeViews)} views")


def predictSubs(daysToGraphForSubs, projectedArray, cumulativeSum, stat):
    f"""Predicts the stat for the days specified."""
    for i in range(len(projectedArray)):
        cumulativeSum += projectedArray[i]
        print(f"[{daysToGraphForSubs[i]}] {stat} Per Day: {projectedArray[i]} | Cumulative {stat}: {math.floor(cumulativeSum)}")
    return cumulativeSum


def chartGrowthOfSubsAndViews(np_views, viewsRegression, np_subs, subsRegression):
    """Reads the CSV into arrays, then lots the arrays 
        and a linear regression and shows the plots"""
    daysToGraphForViews = generateXAxisArray(len(np_views))
    daysToGraphForSubs = generateXAxisArray(len(np_subs))

    figure, axes = plt.subplots(2)
    figure.suptitle("Growth of YouTube Views and Subscribers Over Days")

    graphLinearRegression(axes[0],
                          daysToGraphForViews, np_views, viewsRegression.coef_, viewsRegression.intercept_, "Days", "Views")

    graphLinearRegression(axes[1],
                          daysToGraphForSubs, np_subs, subsRegression.coef_, subsRegression.intercept_, "Days", "Subs")

    plt.show()


def parseData(data_csv_file):
    """Reads the CSV into numpy arrays"""
    delimited_file = pd.read_csv(
        data_csv_file, usecols=[0, 1, 2, 3])
    np_views = np.array(delimited_file['Views'])
    np_subs = np.array(delimited_file['Subscribers'])
    return [np_views, np_subs]


def runLinearRegression(dataToRegress):
    """Generates a linear regression"""
    X = generateXAxisArray(len(dataToRegress))
    X = X.reshape(-1, 1)
    regression = LinearRegression().fit(X, dataToRegress)
    return regression


def graphLinearRegression(axes, xAxis, yAxis1, coefficient, intercept, xLabel, yLabel):
    """Graphs a linear regression"""
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
    """Evaluates a point in the linear regression function"""
    return coefficient * variable + intercept


def generateXAxisArray(len):
    """Generates an array of days of length n."""
    return np.arange(len)


__main__()
