
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import math
from geojson import Point
import geopy.distance

def dist_between_lat_long_points(p1,p2):
    return geopy.distance.vincenty(p1, p2).m

def rotate(origin, point, angle):
    '''Rotate a vector that isn't thorugh origin'''
    new_x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    new_y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    return [new_x, new_y]

earth_radius = 6371000.0
def add_meters_to_lat(lat,dist):
    return lat  + (dist / earth_radius) * (180 / np.pi)
def add_meters_to_long(lat,long,dist):
    return long + (dist / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi/180)
def meters_to_lat(dist):
    return (dist / earth_radius) * (180 / np.pi)
def meters_to_long(lat,dist):
    return (dist / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi/180)

def frame_point_to_rotation(point):
    return yaw_adjust, pitch_adjust

def hover_point_to_lat_long(lat,long,height,yaw,pitch):
    '''
    Given Drone latitude, longitude, height, pitch, and yaw, finds
    latitude and longitude of point on ground in focus of camera view
    '''
    if pitch > 85:
        pitch = 85
    x_dist = np.tan(math.radians(pitch))*height
    if pitch < 0:
        return rotate((flight_points[0][-1],flight_points[1][-1]),(flight_points[0][-1],
                add_meters_to_lat(flight_points[1][-1],x_dist)),math.radians(yaw))
    else:
        return rotate((flight_points[0][-1],flight_points[1][-1]),(flight_points[0][-1],
        add_meters_to_lat(flight_points[1][-1],x_dist)),math.radians(yaw))


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

# Flight record info interpolation
f_record = interp1d(flight_time,record_info)

video = cv2.VideoCapture('flight_footage.mp4')
FOV = 78.8 # DJI Mavic Pro Field of View
height_width_ratio = 3/4 # DJI Mavic Pro Ratio

# Get frames per second and frame size from video
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver) < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_width = video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
else:
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

frame_center = (frame_width/2,frame_height/2)
degree_per_width = FOV / frame_width
degree_per_height = height_width_ratio * FOV / frame_height

# Create interactive plot and limit to flight area
plt.xlim(gps_x_min-.001,gps_x_max+.001)
plt.ylim(gps_y_min-.001,gps_y_max+.001)
plt.ion()
plt.show()

time = 0
flight_points = [[f_x_pos(time)],[f_y_pos(time)]]
i = 1
while True:
    flight_points[0].append(float(f_x_pos(time)))
    flight_points[1].append(float(f_y_pos(time)))
    current_height = np.round(f_height(time),2)
    object_map_points = []
    if f_record(time) == 1: # If camera is recording, show frame
        success, frame = video.read()
        # if not success:
        #     break
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) == 27:
        #     break

    if g_pitch(time) < 0:
        view_pitch = 90 + g_pitch(time)
        dist_to_focus = current_height / np.cos(math.radians(view_pitch))

        drone_focus_point = hover_point_to_lat_long(flight_points[0][-1],flight_points[1][-1],
                                                    current_height,-g_yaw(time),view_pitch)

        '''Insert NN for detection of objects in frame'''
        detections = [(0,0),(frame_width,0),(frame_width,frame_height),(0,frame_height)]

        detection_points = []
        for detection in detections:
            pitch_rotation = (detection[1] - frame_center[1]) * degree_per_height
            yaw_rotation = (detection[0] - frame_center[0]) * degree_per_width
            if (view_pitch + pitch_rotation) < 0:
                yaw_rotation = -yaw_rotation
            point = hover_point_to_lat_long(flight_points[0][-1],flight_points[1][-1],
                    current_height,-g_yaw(time) + yaw_rotation,view_pitch + pitch_rotation)

            detection_points.append(point)

        detection_lines = []
        detection_lines.append(plt.plot([detection_points[0][0],detection_points[1][0]],[detection_points[0][1],detection_points[1][1]],'red'))
        detection_lines.append(plt.plot([detection_points[1][0],detection_points[2][0]],[detection_points[1][1],detection_points[2][1]],'red'))
        detection_lines.append(plt.plot([detection_points[2][0],detection_points[3][0]],[detection_points[2][1],detection_points[3][1]],'red'))
        detection_lines.append(plt.plot([detection_points[3][0],detection_points[0][0]],[detection_points[3][1],detection_points[0][1]],'red'))
        #
        plt.plot(flight_points[0][-2:],flight_points[1][-2:],'blue')
        drone_fp = plt.scatter(drone_focus_point[0],drone_focus_point[1],c='orange')
        plt.pause(.001)
        drone_fp.remove()
        for line in detection_lines:
            line[0].remove()

    # else:
    #     plt.plot(flight_points[0][-2:],flight_points[1][-2:],'blue')
    #     plt.pause(.001)

    # time += 1/fps
    time += .8
    i += 1
# cv2.destroyAllWindows()





#
