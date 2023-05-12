import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import sys


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_PATH)
filename = [f'{ROOT_PATH}/datalog-gitignore/taxi-data/aaaa.csv']
print(filename[0])
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
    plt.plot(group1_resampled.index, group1_resampled, label='Catagory 1')
    plt.plot(group2_resampled.index, group2_resampled, label='Catagory 2')
    plt.xlabel('Time')
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel('15-minute package volume')
    plt.legend()
    plt.show()

def flatData(df):
    group1 = df[df['group'] == 1]
    group2 = df[df['group'] == 2]
    periods = len(group1)
    index = pd.date_range('25/05/2012', periods=periods, freq='T')
    series = pd.Series(range(periods), index=index, name='newdatetime')
    series.to_frame()

    df2 = group1.assign(datetime=series['newdatetime'])
    print(df2)
    # # put the series in data frame group1 in the column 'datetime'
    # group1['group'] = 1
    # #keep the original values 
    # result = group1
    # result.to_csv(f'{ROOT_PATH}/datalog-gitignore/taxi-data/test-manhattan-taxi-20160526-peak-flat.csv', index=False)


    

if __name__ == '__main__':
    df = ReadCSV(filename)
    # flatData(df)
    plot(df)

