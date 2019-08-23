
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import math

def rotate(origin, point, angle):
    new_x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    new_y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    return (new_x, new_y)

earth_radius = 6371000.0
def add_meters_to_lat(lat,dist):
    return lat  + (dist / earth_radius) * (180 / np.pi)
def add_meters_to_long(lat,long,dist):
    return long + (dist / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi/180)

# Read flight log into dataframe
df = pd.read_csv('flight_record.csv',encoding="ISO-8859-1")

# Flight time updates
flight_time = df['OSD.flyTime [s]'].values

# Video recording info
record_info = df['CAMERA_INFO.recordState'].values
record_info = np.where(record_info=='No',0,record_info)
record_info = np.where(record_info=='Starting',1,record_info)

# Find distance boundaries traveled in flight
gps_x_min = np.min(df['OSD.longitude'])
gps_x_max = np.max(df['OSD.longitude'])
gps_y_min = np.min(df['OSD.latitude'])
gps_y_max = np.max(df['OSD.latitude'])

# Drone positional information
x_pos = df['OSD.longitude'].values
y_pos = df['OSD.latitude'].values
alt = df['OSD.altitude [m]'].values
height = df['OSD.height [m]'].values
pitch = df['OSD.pitch'].values
yaw = df['OSD.yaw'].values
roll = df['OSD.roll'].values

# Gimbal positional information
gimbal_pitch = df['GIMBAL.pitch'].values
gimbal_roll = df['GIMBAL.roll'].values
gimbal_yaw = df['GIMBAL.yaw'].values

# Interpolate discrete positional data into continuous functions
f_x_pos = interp1d(flight_time,x_pos)
f_y_pos = interp1d(flight_time,y_pos)
f_height = interp1d(flight_time,height)
f_alt = interp1d(flight_time,alt)
f_pitch = interp1d(flight_time,pitch)
f_roll = interp1d(flight_time,roll)
f_yaw = interp1d(flight_time,yaw)

# Gimbal position interpolation
g_pitch = interp1d(flight_time,gimbal_pitch)
g_roll = interp1d(flight_time,gimbal_roll)
g_yaw = interp1d(flight_time,gimbal_yaw)

# Record info interpolation
f_record = interp1d(flight_time,record_info)

video = cv2.VideoCapture('flight_footage.mp4')
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video.get(cv2.CAP_PROP_FPS)

time = 0

plt.xlim(gps_x_min,gps_x_max)
plt.ylim(gps_y_min,gps_y_max)
plt.ion()
plt.show()

all_pitch = [f_pitch(time)+g_pitch(time)]
all_height = [f_height(time)]
all_yaw = [f_yaw(time)+g_yaw(time)]
points = [[f_x_pos(time)],[f_y_pos(time)]]

focus_points = [[f_x_pos(time)],[f_y_pos(time)]]

i = 1
while True:
    # if f_record(time) == 1: # If camera is recording, show frame
    #     success, frame = video.read()
    #     if not success:
    #         break
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) == 27:
    #         break

    # time += 1/fps
    time += 1
    points[0].append(f_x_pos(time))
    points[1].append(f_y_pos(time))

    current_height = np.round(f_height(time),2)

    # print('Pitch: {}, Yaw: {}, Dist: {}'.format(np.round(g_pitch(time),2),np.round(g_yaw(time),2),current_height))
    if g_pitch(time) < 0:
        x_dist = np.tan(90-g_pitch(time))*current_height
        drone_focus_point = rotate((points[0][-1],points[1][-1]),(points[0][-1],add_meters_to_lat(points[1][-1],x_dist)),math.radians(180-g_yaw(time)))
        focus_points[0].append(drone_focus_point[0])
        focus_points[1].append(drone_focus_point[1])
    else:
        focus_points[0].append(focus_points[0][-1])
        focus_points[1].append(focus_points[1][-1])

    # Find transformation
    pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
    pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Detection points
    a = np.array([[154, 174]], dtype='float32')
    a = np.array([a])

    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(a, h)

    plt.plot(points[0][-2:],points[1][-2:],'blue')
    # plt.plot(focus_points[0][-2:],focus_points[1][-2:],'orange')
    a = plt.scatter(focus_points[0][-1],focus_points[1][-1],c='orange')
    plt.pause(.001)
    a.remove()

    i += 1
# cv2.destroyAllWindows()





#
