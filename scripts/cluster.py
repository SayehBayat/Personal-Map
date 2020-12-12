from geopy.distance import great_circle
import os
import pandas as pd
import datetime

# Adapted from https://github.com/jayachithra/T-DBSCAN
# Original paper: "T-DBSCAN: A Spatiotemporal Density Clustering for GPS Trajectory Segmentation"

def T_DBSCAN(df, CEps=200, Eps=100, MinPts=3):
    C = 0
    Cp = {}
    UNMARKED = 777777

    df['cluster'] = UNMARKED
    df['visited'] = 'Not visited'
    MaxId = -1

    for index, P in df.iterrows():
        if index > MaxId:

            df.at[index, 'visited'] = 'visited'
            # search for continuous density-based neighbours N
            N = getNeighbors(P, CEps, Eps, df, index)
            MaxId = index
            # create new cluster
            if len(N) > MinPts:
                C = C + 1
                # expand the cluster
            Ctemp, MaxId = expandCluster(P, N, CEps, Eps, MinPts, MaxId, df, index)
            if C in Cp:
                Cp[C] = Cp[C] + Ctemp
            else:
                Cp[C] = Ctemp

    print("Clusters identified...")
    Cp = mergeClusters(Cp)  # merge clusters
    df = updateClusters(df, Cp)  # update df

    return df


# Retrieve neighbors
def getNeighbors(P, CEps, Eps, df, p_index):
    neighborhood = []
    center_point = P

    for index, point in df.iterrows():
        if index > p_index:
            distance = great_circle((center_point['latitude'], center_point['longitude']),
                                    (point['latitude'], point['longitude'])).meters
            if distance < Eps:
                neighborhood.append(index)
            elif distance > CEps:
                break

    return neighborhood


# cluster expanding
def expandCluster(P, N, CEps, Eps, MinPts, MaxId, df, p_index):
    Cp = []
    N2 = []

    Cp.append(p_index)

    for index in N:
        point = df.loc[index]
        df.loc[index]['visited'] = 1
        if index > MaxId:
            MaxId = index
        N2 = getNeighbors(point, CEps, Eps, df, index)  # find neighbors of neighbors of core point P
        if len(N2) >= MinPts:  # classify the points into current cluster based on definitions 3,4,5
            N = N + N2
        if index not in Cp:
            Cp.append(index)

    return Cp, MaxId


# merge clusters
def mergeClusters(Cp):
    Buffer = {}

    print("Merging...")
    for idx, val in Cp.items():

        if not Buffer:  # if buffer is empty add first item by default
            Buffer[idx] = val

        else:  # compare last item in the buffer with Cp
            if max(Buffer[list(Buffer.keys())[-1]]) <= min(Cp[idx]):  # new cluster = new Buffer entry
                Buffer[(list(Buffer.keys())[-1]) + 1] = Cp[idx]
            else:  # merge last item in the buffer with Cp
                Buffer[list(Buffer.keys())[-1]] += Cp[idx]

    return Buffer


# update dataframe
def updateClusters(df, Cp):
    for idx, val in Cp.items():
        for index in val:
            df.at[index, 'cluster'] = idx

    return df

def get_user_df_sorted(df,uid):
    df1 = df[df["uid"] == uid]
    df1.reset_index(drop=True, inplace=True)
    df1["timestamp"] = df1['CreatedTime(Eastern Time)'].apply(timestr_to_timestamp)
    df1 = df1.rename(columns={"Longitude": "longitude", "Latitude": "latitude"})
    return df1.sort_values(by=['timestamp'])

def timestr_to_timestamp(date_time_str):
    return datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S').timestamp()

if __name__ == '__main__':
    os.chdir('..')
    #cdf = pd.read_csv("./data/cluster.csv")
    df = pd.read_csv("./data/dementia_data.csv")
    print(df.uid.unique())
    df1 = get_user_df_sorted(df, "GPS2051")
    cdf = T_DBSCAN(df1)
    cdf.to_csv("./data/cluster_2051.csv")
    #df1 = df[df["uid"] == "GPS5398"]
    #df1.reset_index(drop=True, inplace=True)
    #cdf.groupby(cdf["cluster"]).count()
    print(cdf.cluster.unique())
