from geopy.distance import great_circle
import os
import pandas as pd
import datetime

def get_trips_stops(df):
    trips = []
    stops = []
    for k, v in df.groupby((df['visited'].shift() != df['visited']).cumsum()):
        #print(f'[group {k}]')
        #print(v)
        if v.visited.tolist()[0] == "visited":
            trips.append(v)
        else:
            stops.append(v)
    return trips, stops
def create_trip_df(trips):
    tdf = {'eventTimeStart': [], 'latStart': [], 'lonStart': [], 'latEnd': [], 'lonEnd': [], 'len': []}
    for trip in trips:
        print(len(trip))
        if len(trip) > 2:
            tdf["eventTimeStart"].append(trip['CreatedTime(Eastern Time)'].tolist()[0])
            tdf["latStart"].append(trip['latitude'].tolist()[0])
            tdf["lonStart"].append(trip['longitude'].tolist()[0])
            tdf["latEnd"].append(trip['latitude'].tolist()[-1])
            tdf["lonEnd"].append(trip['longitude'].tolist()[-1])
            tdf["len"].append(len(trip))
    return pd.DataFrame(tdf)
if __name__ == '__main__':
    os.chdir('..')
    df = pd.read_csv("./data/cluster_3797.csv")
    trips, stops = get_trips_stops(df)
    tdf = create_trip_df(trips[2:])
    tdf.to_csv("./data/trips_2051.csv")
    print(tdf.len)
    #print(len(stops))
