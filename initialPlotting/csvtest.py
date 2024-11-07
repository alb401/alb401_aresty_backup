import csv
import pandas as pd

temps = [] # Temperatures
res = [] # Resistivities

df = pd.read_csv('1_RT_EIO4732.csv')
# print(df.to_string()) 
print(pd.options.display.max_rows)


# with open('1_RT_EIO4732.csv', newline='') as csvfile:

#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

#     for row in spamreader: # gets read in as a 1 element list
#         x = row[0].split(",") # splits at the comma, becomes 2 element list
#         temps.append(x[0])
#         res.append(x[1])

# -- Prints out temps and resitivities for whole csv file -- 
# for i in temps: 
#     print("temp: " + i + " || resistivity: " + res[temps.index(i)])

