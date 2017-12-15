'''data downloaded from:
   https://www.kaggle.com/sudalairajkumar/indian-startup-funding

   Problem statement: Demonstration of K-Means clustering on the given dataset
   CS7018 Anurag Banerjee (17071003)
'''

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as Mpl
import datetime as dt


# Global variables
X = [[]]
Y = [[]]
nc = 0
score = []
pca = 0
pca_d = 0
pca_c = 0
kmeansClust = 0
maxCluster = 0


def readData():
    '''
    This function reads in the dataser and performs minor preprocessing
    :return: None
    '''
    global X, Y
    dateparser = lambda x: pd.datetime.strptime(x, '%d-%m-%y')  # regex to parse the date data
    variables = pd.read_csv('startup_funding.csv', parse_dates=['Date'], date_parser=dateparser)
    variables.fillna(0, inplace=True)   # replace NA cells with the default value of zero
    variables['Date'] = (pd.to_datetime(variables['Date']) - dt.datetime(1970, 1, 1)).dt.total_seconds()
    Y = variables[['AmountInUSD']]
    X = variables[['Date']]
    print 'Data load complete...'


def getClusters():
    '''
    This function performs the KMeans clustering
    :return: None
    '''
    global nc, score, kmeansClust, Y, maxCluster
    maxScore = -float("inf")
    nc = range(1, 20)   # Number of clusters
    kmeansClust = [KMeans(n_clusters=i) for i in nc]
    # the scores below are used to generate the elbow curve
    score = [kmeansClust[i].fit(Y).score(Y) for i in range(len(kmeansClust))]

def plotElbowCurve():
    '''
    This function plots an elbow curve of the score calculated versus the number of clusters
    :return: None
    '''
    print 'Plotting the elbow curve...'
    Mpl.figure('Elbow Curve')
    Mpl.xlabel('Number of clusters')
    Mpl.ylabel('Score')
    Mpl.title('Elbow Curve')
    Mpl.plot(nc, score, 'bx-')
    Mpl.show()


def principalCompDecom():
    '''
    This function performs PCA on amount in US dollar
    :return: None
    '''
    global pca, pca_c, pca_d, X, Y
    print 'Performing PCA based decomposition...'
    pca = PCA(n_components=1).fit(Y)
    pca_d = pca.transform(Y)
    pca_c = pca.transform(X)


def showClusters():
    '''
    This function plots the scatter plot for the color coded clusters
    :return: None
    '''
    global X, Y, pca_c, pca_d
    print 'Plotting scatter plot of clusters...'
    kmeans = KMeans(n_clusters=5)   # the value 5 came from observing the elbow curve
    kmeansoutput = kmeans.fit(Y)
    Mpl.figure('5 Cluster K-Means')
    Mpl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
    Mpl.xlabel('Seconds since UNIX Epoch')
    Mpl.ylabel('AmountInUSD')
    Mpl.title('5 Cluster K-Means')
    Mpl.show()


if __name__ == '__main__':
    print 'Loading data...'
    readData()
    getClusters()
    plotElbowCurve()
    principalCompDecom()
    showClusters()
    print 'Successfully completed operation.'
