import os
import pandas as pd

# print("\n\n" + os.getcwd())

directory = os.path.join(os.getcwd(), "data")
print("\n\n" + os.getcwd())

for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           newpath = directory + '/' + file
        #    f = open(newpath, 'r')
        #    print(f.name)
           df = pd.read_csv(newpath)
           print(df.head())