# Needed libraries
import time
from datetime import date
import math
import os
import pandas as pd
import geohash

# Adapted from https://github.com/carlosbkm/car-destination-prediction/blob/master/features-preparation.ipynb


def date_extractor(date_str, b, minutes_per_bin):
    # Takes a datetime object as a parameter
    # and extracts and returns a tuple of the form: (as per the data specification)
    # (time_cat, time_num, time_cos, time_sin, day_cat, day_num, day_cos, day_sin, weekend)
    # Split date string into list of date, time

    d = date_str.split()

    # safety check
    if len(d) != 2:
        return tuple([None, ])

    # TIME (eg. for 16:56:20 and 15 mins per bin)
    # list of hour,min,sec (e.g. [16,56,20])
    time_list = [int(t) for t in d[1].split(':')]

    # safety check
    if len(time_list) != 3:
        return tuple([None, ])

    # calculate number of minute into the day (eg. 1016)
    num_minutes = time_list[0] * 60 + time_list[1]

    # Time of the start of the bin
    time_bin = num_minutes / minutes_per_bin  # eg. 1005
    hour_bin = num_minutes / 60  # eg. 16
    min_bin = (time_bin * minutes_per_bin) % 60  # eg. 45

    # get time_cat
    hour_str = str(hour_bin) if hour_bin / 10 > 0 else "0" + str(hour_bin)  # eg. "16"
    min_str = str(min_bin) if min_bin / 10 > 0 else "0" + str(min_bin)  # eg. "45"
    time_cat = hour_str + ":" + min_str  # eg. "16:45"

    # Get a floating point representation of the center of the time bin
    time_num = (hour_bin * 60 + min_bin + minutes_per_bin / 2.0) / (60 * 24)  # eg. 0.7065972222222222

    time_cos = math.cos(time_num * 2 * math.pi)
    time_sin = math.sin(time_num * 2 * math.pi)

    # DATE
    # Parse year, month, day
    date_list = d[0].split('-')
    d_obj = date(int(date_list[0]), int(date_list[1]), int(date_list[2]))
    day_to_str = {0: "Monday",
                  1: "Tuesday",
                  2: "Wednesday",
                  3: "Thursday",
                  4: "Friday",
                  5: "Saturday",
                  6: "Sunday"}
    day_of_week = d_obj.weekday()
    day_cat = day_to_str[day_of_week]
    day_num = (day_of_week + time_num) / 7.0
    day_cos = math.cos(day_num * 2 * math.pi)
    day_sin = math.sin(day_num * 2 * math.pi)

    year = d_obj.year
    month = d_obj.month
    day = d_obj.day

    weekend = 0
    # check if it is the weekend
    if day_of_week in [5, 6]:
        weekend = 1

    return (year, month, day, time_cat, time_num, time_cos, time_sin, day_cat, day_num, day_cos, day_sin, weekend)


def data_cleaner(zipped_row):
    # takes a tuple (row,g,b,minutes_per_bin) as a parameter and returns a tuple of the form:
    # (time_cat, time_num, time_cos, time_sin, day_cat, day_num, day_cos, day_sin, weekend,geohash)
    row = zipped_row[0]
    g = zipped_row[1]
    b = zipped_row[2]
    minutes_per_bin = zipped_row[3]
    # The indices of trip-start datetime, latitude start, longitude start, latitude end and longitude end respectively
    indices = (0, 1, 2, 3, 4)

    # safety check: make sure row has enough features
    if len(row) < 5:
        return None

    # extract day of the week and hour
    date_str = row[indices[0]]
    clean_date = date_extractor(date_str, b, minutes_per_bin)
    # get geo hash
    lat_start = float(row[indices[1]])
    lon_start = float(row[indices[2]])
    lat_end = float(row[indices[3]])
    lon_end = float(row[indices[4]])
    location_start = None
    location_end = None
    location_start = geohash.encode(lat_start, lon_start, g)
    location_end = geohash.encode(lat_end, lon_end, g)
    x_start = math.cos(lat_start) * math.cos(lon_start)
    y_start = math.cos(lat_start) * math.sin(lon_start)
    z_start = math.sin(lat_start)

    return tuple(list(clean_date) + [x_start] + [y_start] + [z_start] + [location_start] + [location_end])

def create_df(df, g=9, b=12):
    df = df[['eventTimeStart', 'latStart', 'lonStart', 'latEnd', 'lonEnd']]

    minutes_per_bin = int((24 / float(b)) * 60)
    rows = df.apply(lambda row: (list(row.values), g, b, minutes_per_bin), axis=1)

    transformed_data = []
    for row in rows:
        transformed_data.append(data_cleaner(row))
    df = pd.DataFrame(transformed_data, columns=['year', 'month', 'day', 'time_cat', 'time_num', 'time_cos',
                                                 'time_sin', 'day_cat', 'day_num', 'day_cos', 'day_sin', 'weekend',
                                                 'x_start', 'y_start', 'z_start','location_start', 'location_end'])
    return df

if __name__ == '__main__':
    os.chdir('..')
    df = pd.read_csv("./data/trips.csv")
    g = 9 #geohash length, a 1.2km x 609.4m square area
    b = 12 # number of time bins per day
    tdf = create_df(df, g=9, b=12)
    print(tdf.head())
    exit()

