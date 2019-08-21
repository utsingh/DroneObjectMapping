
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d

df = pd.read_csv('flight_record.csv')
print(df.columns)
a

subtract_time = df['Time(seconds)'][0]

gps_x_min = np.min(df['Longitude'])
gps_x_max = np.max(df['Longitude'])
gps_y_min = np.min(df['Latitude'])
gps_y_max = np.max(df['Latitude'])

gps_position = df.loc[:,'Latitude':'Longitude']
x_pos = gps_position['Longitude']
y_pos = gps_position['Latitude']
alt = df.loc[:,'Altitude(feet)']
pitch = df.loc[:,'Pitch']
yaw = df.loc[:,'Yaw(360)']

time = []
temp = df['Time(seconds)']
for i in range(len(temp)):
    time.append(temp[i] - subtract_time)

f_x_pos = interp1d(time, x_pos)
f_y_pos = interp1d(time, y_pos)
f_alt =interp1d(time,alt)
f_pitch =interp1d(time,pitch)
f_yaw =interp1d(time,yaw)
a

water = gpd.read_file('/Users/samclymer/Desktop/water/hydrology.shp')
water.plot()
plt.xlim(gps_x_min-.005,gps_x_max+.005)
plt.ylim(gps_y_min-.005,gps_y_max+.005)
# plt.show()

fps = 30
video = cv2.VideoCapture('flight_footage.mp4')
time = 0

points = [[],[]]
points[0].append(f_x_pos(time))
points[1].append(f_y_pos(time))

plt.ion()
plt.show()
while True:
    # success, frame = video.read()
    # if not success:
        # break
    time += 1/30
    # print('Time: {} Seconds'.format(np.round(time,2)))
    points[0].append(f_x_pos(time))
    points[1].append(f_y_pos(time))

    x_dist = np.arctan(f_pitch(time))

    plt.plot(points[0][-2:],points[1][-2:],'orange')
    plt.pause(.001)
    # frame = cv2.resize(frame,(int(frame.shape[1]/5),int(frame.shape[0]/5)))
    # cv2.imshow('frame',frame)

#     if cv2.waitKey(1) == 27:
#         break
# cv2.destroyAllWindows()







































#
