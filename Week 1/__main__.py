import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from customplot import *

# create a function for plotting a dataframe with string columns and numeric values

def plot_dataframe(df, y_label):
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

    ax = df.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.attributes, rotation=75); #Notice the ; (remove it and see what happens !)
    plt.show()

if __name__ == "__main__":
    #ingest data
    cnx = sqlite3.connect('database.sqlite')
    df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

    #exploring data
    print(df.columns)
    print(df.describe().transpose())

    #data cleaning: handling missing data
    #is any row null?
    print(df.isnull().any().any(), df.shape)

    #data points in each column that are null
    print(df.isnull().sum(axis=0))

    #fixing null values by deleting them

    #Take initial # of rows
    rows = df.shape[0]
    #Drop the NULL rows
    df = df.dropna()

    #Check if all NULLS are gone?
    print(df.isnull().any().any(), df.shape)

    #How many rows with NULL values?
    print(rows - df.shape[0])

    #Shuffle the rows of df so we get a distributed sample when we display top few rows
    df = df.reindex(np.random.permutation(df.index))

    #predicting 'overall_rating' of a player
    print(df.head(5))
    print(df[:10][['penalties', 'overall_rating']])

    #feature correlation analysis
    #we will check if 'penalties' is correlated to 'overall_rating'
    #using Pearson's correlation coefficient
    print(df['overall_rating'].corr(df['penalties']))
    #Pearson goes from -1 to +1. A value of 0 would have told there is
    # no correlation. A value of 0.39 shows some correlation, although it
    # could be stronger

    #Create a list of potential features that you want to measure
    #correlation with
    potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']
    # check how the features are correlated with the overall ratings
    for f in potentialFeatures:
        related = df['overall_rating'].corr(df[f])
        print("%s: %f" % (f, related))

    #Data visualization
    cols = ['potential', 'crossing', 'finishing', 'heading_accuracy',
            'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
            'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
            'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
            'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
            'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
            'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
            'gk_reflexes']

    # create a list containing Pearson's correlation between 'overall_rating' with each column in cols
    correlations = [df['overall_rating'].corr(df[f]) for f in cols]
    print(len(cols), len(correlations))

    #create a dataframe using cols and correlations
    df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations})
    # let's plot above dataframe using the function we created
    plot_dataframe(df2, 'Player\'s Overall Rating')

    #Analysis of findings
    #Clustering players into similar groups

    # Define the features you want to use for grouping players
    select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']
    # Generate a new dataframe by selecting the features you just defined
    df_select = df[select5features].copy(deep=True)
    print(df_select.head())

    #Perform KMeans Clustering
    #Perform scaling on the dataframe containing the features
    data = scale(df_select)
    #Define number of clusters
    noOfClusters = 4
    #Train a model
    model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)
    print(90 * '_')
    print("\nCount of players in each cluster")
    print(90 * '_')
    print(pd.value_counts(model.labels_, sort=False))

    # Create a composite dataframe for plotting
    # ... Use custom function declared in customplot.py (which we imported at the beginning of this notebook)
    P = pd_centers(featuresUsed=select5features, centers=model.cluster_centers_)
    print(P)
    parallel_plot(P)

    #Analysis of Findings
    #...




