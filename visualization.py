import numpy as np
import pandas as pd
import pylab

import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data, data_preprocessing, DATA_PATH

def plot_category_distrib(data):
    """"""
    distrib = data.Category.value_counts()
    print(distrib)
    plt.figure(figsize=(10,5))
    sns.barplot(distrib.index, distrib.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title('Crime Category distribution')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Crime Category', fontsize=12)
    plt.show()

def plot_resolution_distrib(data):
    """"""
    distrib = data.Resolution.value_counts()
    print(distrib)
    plt.figure(figsize=(10,5))
    sns.barplot(distrib.index, distrib.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title('Crime Resolution distribution')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Resolution', fontsize=12)
    plt.show()

def plot_PdDistrict_distrib(data):
    """"""
    distrib = data.PdDistrict.value_counts()
    print(distrib)
    plt.figure(figsize=(10,5))
    sns.barplot(distrib.index, distrib.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title('Police Department District distribution')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Police Department District', fontsize=12)
    plt.show()

def plot_week_distrib(data):
    """"""
    distrib = data.DayOfWeek.value_counts()
    print(distrib)
    plt.figure(figsize=(10,5))
    sns.barplot(distrib.index, distrib.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title('Number of crime by day of the week distribution')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Day of the week', fontsize=12)
    plt.show()

def plot_dayTime_distrib(data):
    data['DayOfWeek'] = data.Dates.dt.dayofweek
    data['Hour'] = data.Dates.dt.hour
    data['Month'] = data.Dates.dt.month
    data['Year'] = data.Dates.dt.year
    data['DayOfMonth'] = data.Dates.dt.day

    plt.figure(figsize=(10,5))
    plt.plot(data.groupby('Hour').size(), 'ro-')
    plt.title('Evolution of the number of crimes in a day')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Hours', fontsize=12)
    plt.xticks(np.arange(0, 24, 1))
    plt.show()

def plot_top9_dayTime_distrib(data):
    data['DayOfWeek'] = data.Dates.dt.dayofweek
    data['Hour'] = data.Dates.dt.hour
    data['Month'] = data.Dates.dt.month
    data['Year'] = data.Dates.dt.year
    data['DayOfMonth'] = data.Dates.dt.day

    pylab.rcParams['figure.figsize'] = (14.0, 8.0)

    larceny = data[data['Category'] == "LARCENY/THEFT"]
    assault = data[data['Category'] == "ASSAULT"]
    drug = data[data['Category'] == "DRUG/NARCOTIC"]
    vehicle = data[data['Category'] == "VEHICLE THEFT"]
    vandalism = data[data['Category'] == "VANDALISM"]
    warrants = data[data['Category'] == "WARRANTS"]
    burglary = data[data['Category'] == "BURGLARY"]
    other = data[data['Category'] == "OTHER OFFENSES"]
    nonCriminal = data[data['Category'] == "NON-CRIMINAL"]

    with plt.style.context('fivethirtyeight'):

        ax1 = plt.subplot2grid((3,3), (0, 0))
        ax1.plot(larceny.groupby('Hour').size(), 'o-')
        ax1.set_title ('Larceny/Theft')

        ax2 = plt.subplot2grid((3,3), (0, 1))
        ax2.plot(assault.groupby('Hour').size(), 'o-')
        ax2.set_title ('Assault')

        ax3 = plt.subplot2grid((3,3), (0, 2))
        ax3.plot(drug.groupby('Hour').size(), 'o-')
        ax3.set_title ('Drug/Narcotic')

        ax4 = plt.subplot2grid((3,3), (1, 0))
        ax4.plot(vehicle.groupby('Hour').size(), 'o-')
        ax4.set_title ('Vehicle')

        ax5 = plt.subplot2grid((3,3), (1, 1))
        ax5.plot(vandalism.groupby('Hour').size(), 'o-')
        ax5.set_title ('Vandalism')

        ax6 = plt.subplot2grid((3,3), (1, 2))
        ax6.plot(warrants.groupby('Hour').size(), 'o-')
        ax6.set_title ('Warrants')

        ax7 = plt.subplot2grid((3,3), (2, 0))
        ax7.plot(burglary.groupby('Hour').size(), 'o-')
        ax7.set_title ('Burglary')

        ax8 = plt.subplot2grid((3,3), (2, 1))
        ax8.plot(other.groupby('Hour').size(), 'o-')
        ax8.set_title ('Other')

        ax9 = plt.subplot2grid((3,3), (2, 2))
        ax9.plot(nonCriminal.groupby('Hour').size(), 'o-')
        ax9.set_title ('Non-criminal')

        pylab.gcf().text(0.5, 0.95,
                        'San Franciso TOP 9 Crime Occurence by Hour',
                         horizontalalignment='center',
                         verticalalignment='top',
                         fontsize = 28)

    plt.tight_layout(2)
    plt.show()

def plot_full_crime_evolution(data):

    data['Month'] = data.Dates.dt.month
    data['Year'] = data.Dates.dt.year

    pylab.rcParams['figure.figsize'] = (16.0, 5.0)
    yearMonth = data.groupby(['Year','Month']).size()
    ax = yearMonth.plot(lw=2)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 1))
    ax.get_xaxis().tick_bottom()
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Year (2003-2015)', fontsize=12)
    plt.title('Crime occurences by year', fontsize=24)
    plt.show()

def plot_year_distribution(data):

    # color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b', '']
    crimes = ["LARCENY/THEFT","ASSAULT","DRUG/NARCOTIC","VEHICLE THEFT","VANDALISM",
              "WARRANTS","BURGLARY","OTHER OFFENSES","NON-CRIMINAL"]

    data['Year'] = data.Dates.dt.year

    larceny = data[data['Category'] == crimes[0]]
    assault = data[data['Category'] == crimes[1]]
    drug = data[data['Category'] == crimes[2]]
    vehicle = data[data['Category'] == crimes[3]]
    vandalism = data[data['Category'] == crimes[4]]
    warrants = data[data['Category'] == crimes[5]]
    burglary = data[data['Category'] == crimes[6]]
    other = data[data['Category'] == crimes[7]]
    nonCriminal = data[data['Category'] == crimes[8]]

    pylab.rcParams['figure.figsize'] = (16.0, 10.0)

    years = data.groupby('Year').size().keys()
    occursByYear = data.groupby('Year').size().get_values()

    # Linear normalized plot for 6 top crimes
    ax = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=3)

    y = np.empty([9,13])
    h = [None]*9
    y[0] = larceny.groupby('Year').size().get_values()
    y[1] = assault.groupby('Year').size().get_values()
    y[2] = drug.groupby('Year').size().get_values()
    y[3] = vehicle.groupby('Year').size().get_values()
    y[4] = vandalism.groupby('Year').size().get_values()
    y[5] = warrants.groupby('Year').size().get_values()
    y[6] = burglary.groupby('Year').size().get_values()
    y[7] = other.groupby('Year').size().get_values()
    y[8] = nonCriminal.groupby('Year').size().get_values()

    for i in range(0,9):
        h[i] = ax.plot(years, y[i],'o-', lw=2)

    ax.set_ylabel("Crime occurences by year")

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+2, 1))
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Year (2003-2015)', fontsize=12)

    ax.legend((item[0] for item in h),
               crimes,
               bbox_to_anchor=(0.87, 1), loc=2, borderaxespad=0., frameon=False)

    pylab.gcf().text(0.5, 0.95,
                'Crime Occurence by Year',
                horizontalalignment='center',
                verticalalignment='top',
                 fontsize = 28)
    plt.show()


def plot_maps(data):
    mapdata = np.loadtxt(DATA_PATH + "/sf_map.txt")
    asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

    lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
    clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

    #Get rid of the bad lat/longs
    data['Xok'] = data[data.X<-121].X
    data['Yok'] = data[data.Y<40].Y
    data = data.dropna()
    data = data[1:300000] #Can't use all the data and complete within 600 sec :(

    g= sns.FacetGrid(data, col="Category", col_wrap=6, size=5, aspect=1/asp)

    for ax in g.axes:
        ax.imshow(mapdata, cmap=plt.get_cmap('gray'),
                  extent=lon_lat_box,
                  aspect=asp)
    g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

    plt.savefig('./visualization/category_density_plot.png')
    plt.show()

    # #Do a larger plot with prostitution only
    # plt.figure(figsize=(20,20*asp))
    # ax = sns.kdeplot(trainP.Xok, trainP.Yok, clip=clipsize, aspect=1/asp)
    # ax.imshow(mapdata, cmap=plt.get_cmap('gray'),
    #               extent=lon_lat_box,
    #               aspect=asp)
    # # plt.savefig('prostitution_density_plot.png')

def main():
    train, test = load_data()
    # print(list(train))
    # plot_week_distrib(train)
    # time(train)
    # plot_dayTime_distrib(train)
    # plot_top9_dayTime_distrib(train)
    # plot_full_crime_evolution(train)
    # plot_year_distribution(train)
    # plot_maps(train)
    plot_resolution_distrib(train)
    print(train.Resolution.unique())

if __name__ == '__main__':
    main()
