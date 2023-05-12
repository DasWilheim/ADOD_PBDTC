import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_PATH)
filename = [f'{ROOT_PATH}/datalog-gitignore/taxi-data/20160525.csv']

def ReadCSV(filename):
    df = pd.read_csv(filename[0], skiprows=[0], header=None, names=['datetime', 'value1', 'value2', 'group'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    return df

def plot(df):
    # Set the 'datetime' column as the index
    df.set_index('datetime', inplace=True)

    # Split the data into two groups based on the 'group' column
    group1 = df[df['group'] == 1]
    group2 = df[df['group'] == 2]

    # Resample the data to a time frequency (e.g., every minute) and count the number of rows in each group within each time interval
    group1_resampled = group1.resample('15T').size()
    group2_resampled = group2.resample('15T').size()

    # Create a line plot of the number of rows in each group over time
    plt.plot(group1_resampled.index, group1_resampled, label='Group 1')
    plt.plot(group2_resampled.index, group2_resampled, label='Group 2')
    plt.xlabel('Time')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of rows')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = ReadCSV(filename)
    # flatData(df)
    # plot(df)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)

    # Resample the data to 15-minute intervals and sum the values in the value1 column
    orders_15min = df['value1'].resample('15T').size()
    
    # Plot the resampled data
    orders_15min.plot()
    plt.show()
    #now create a new csv file with just a single colum with the orders_15min series
    # orders_15min.to_csv(f'{ROOT_PATH}/datalog-gitignore/taxi-data/20160526-flat.csv', index=False)
