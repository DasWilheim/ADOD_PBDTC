import pandas as pd
from csv import writer

df1_list = []
df2_list = []

df1 = pd.read_csv('data_with_travel_time.csv', index_col=None, header=None)
df2 = pd.read_csv('data_with_travel_time1.2.csv', index_col=None, header=None)
df1_list.append(df1) # here .T means you are transposing the dataframe
df2_list.append(df2)

# print(df1_list[0][:10])
# print(df2_list[0][:22])

new_list = df1_list[0][1:10]
new_list2 = df2_list[0][1:22]
# print(new_list)
df_total = []

for i in range(1, 22):
    df_new =df1[((i-1)*9+1):i*9+1]

    df_total.append(df_new)

    df_new =df2[((i-1)*21+1):i*21+1]
    df_total.append(df_new)

df_total = pd.concat(df_total, ignore_index=True)
print(df_total[0:41])

#now write the new dataframe to a new csv file
df_total.to_csv('data_with_travel_time_total.csv', index=False, header=False)

# for i in range(20)

 
# # List that we want to add as a new row
# List = [6, 'William', 5532, 1, 'UAE']
 
# # Open our existing CSV file in append mode
# # Create a file object for this file
# with open('event.csv', 'a') as f_object:
 
#     # Pass this file object to csv.writer()
#     # and get a writer object
#     writer_object = writer(f_object)
 
#     # Pass the list as an argument into
#     # the writerow()
#     writer_object.writerow(List)
 
#     # Close the file object
#     f_object.close()