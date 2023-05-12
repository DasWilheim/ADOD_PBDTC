import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# Define the time range

def create_datetime_list(start_time, samples):
    factor = (1440/samples)/2
    time_step = datetime.timedelta(minutes=1)
    timedeltas = []
    for i in range(samples):

        ##############    REPLACE THE STEP FUNCTION BELOW WITH YOUR OWN    ##############
        # step = (np.cos((2 * np.pi/samples)*i) + 2)*factor
        step = 1440/samples
        time_step = datetime.timedelta(minutes=step)
        timedeltas.append(time_step)

    start_date = start_time
    datetime_list = []
    for i in range(samples):
        start_date += timedeltas[i]
        datetime_list.append(start_date)

    return datetime_list

def create_random_list(df):
    # get the first row
    first_row = df.iloc[[0]]

    # get all other rows except the first one
    other_rows = df.iloc[:]

    first_col = other_rows.iloc[:, [0]]

    # get all other columns except the first one
    other_cols = other_rows.iloc[:, 1:]
   
    # shuffle the other rows randomly
    shuffled_cols = other_cols.sample(frac=1).reset_index(drop=True)
    
    result1 = pd.concat([first_col, shuffled_cols], axis=1)

    # combine the first row with the shuffled other rows
    result = pd.concat([first_row, result1])

    # print the resulting dataframe
    return result


def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def write_csv(filename, df):
    df.to_csv(filename, index=False)

def reduce_length(df, factor):
    df = df.iloc[:int(len(df)/factor)]
    return df   


def replace_datetime_column(df, datetime_list):
    df.iloc[:, 0] = datetime_list
    return df

def randomly_assign_priority(df, ratio):
    #with a ratio between 1 and 2 
    prob_prio1 = 1 / (ratio + 1)
    prob_prio2 = 1 - prob_prio1
    
    # randomly assign priority based on the probabilities
    df['prio'] = np.random.choice([1, 2], len(df), p=[prob_prio2, prob_prio1])
    return df



def to_fraction(x, eps=1e-6):
    a, b = math.floor(x), 1
    if abs(x - a) < eps:
        return (a, b)
    p, q = a, 1
    while True:
        p += 1
        new_x = 1 / (x - p)
        new_a = math.floor(new_x)
        new_b = new_a * q + b
        if abs(new_x - new_a) < eps:
            return (new_b, new_a)
        p, q, x, a, b = new_a * p - q, new_b, new_x, new_a, new_b


def systematically_assign_priority(df, fraction):
    #with a ratio between 1 and 2 
    # I have a array fractions that is (6, 5) or (7, 5) 
    # Now I want to assign priority 1 to 8/13 of the rows and priority 2 to 5/13 of the rows
    # I want to do this systematically, so that the first 8 rows get priority 1 and the next 5 rows get priority 2
    # and so on
    # I have a dataframe with 100 rows
    # I want to assign priority 1 to 8 rows and priority 2 to 5 rows
    for i in range(len(df)):
        if i % (fraction[0] + fraction[1]) < fraction[0]:
            df['prio'][i] = int(1)
        else:
            df['prio'][i] = int(2)


    return df

def add_priotity_column(df, priority):
    df['prio'] = priority
    return df

def reverse_csv_file(df):

    # Reverse the rows in the dataframe
    df_reversed = df.iloc[::-1]

    # Save the reversed dataframe back to a CSV file
    return df_reversed

def concatinate_and_order_csv_files(df1, df2):
    df = pd.concat([df1, df2])
    df = df.sort_values(by=['ptime'])
    return df

def plot(df):
    # Set the 'datetime' column as the index
    df['ptime'] = pd.to_datetime(df['ptime'])
    df.set_index('ptime', inplace=True)
    # Split the data into two groups based on the 'group' column
    group1 = df[df['prio'] == 1]
    group2 = df[df['prio'] == 2]
    print(len)

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

# # creating multiple datafiles
fractions = np.array([(1, 19), (1, 9), (3, 17), (1,4), (1, 3), (3, 7), (7, 13), (2, 3), (9, 11), (1, 1), (11, 9), (3, 2), (13, 7), (7, 3), (3, 1), (4, 1), (17,3), (9, 1), (19, 1)])
versions = ['05' , '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
priority = [ '3.1' , '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '4.0', '4.1', '4.2', '4.3', '4.4', '4.5']

# creating single datafile
# fractions = np.array([(0.2, 1)])
# versions = np.array([0.8])


for j in range(len(priority)):
    for i in range(len(versions)):
        print(fractions[i])
        df = read_csv('20160525-flat1.6.csv')
        df = create_random_list(df)
        df = replace_datetime_column(df, create_datetime_list(datetime.datetime(2023, 3, 13, 0, 0, 0), (len(df))))
        df = systematically_assign_priority(df, fractions[i])
        group1 = df[df['prio'] == 1]
        group2 = df[df['prio'] == 2]
        print(len(group1))
        print(len(group2))
        write_csv(f'data_version_priority2/20160525-flat{versions[i]}-{priority[j]}.csv', df)

        # plot(df)

# df = add_priotity_column(df, 2)

# df_concat = concatinate_and_order_csv_files(df1, df2)


# plot(df)