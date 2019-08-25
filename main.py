
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import math
from geojson import Point
import geopy.distance

def scale_vector(vector,scalar):
    vector = np.array(vector)
    temp = np.array([[0,0],[vector[1,0]-vector[0,0],vector[1,1]-vector[0,1]]])
    temp = scalar * temp / np.linalg.norm(temp)
    temp[0] = vector[0]
    temp[1] += vector[0]
    return temp

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

def hover_point_to_lat_long(long,lat,height,yaw,pitch,x_pos):
    '''
    Given Drone latitude, longitude, height, pitch, and yaw, finds
    latitude and longitude of point on ground

    ### TODO: Fix this formula to correctly take into account curvature in camera frame ###
    '''
    if pitch > 85:
        pitch = 85
    if pitch == 0:
        pitch = .00000001
    dist_to_focus = height / np.cos(math.radians(pitch))
    x_dist = np.tan(math.radians(pitch))*height
    # temp = rotate((long,lat),(long,add_meters_to_lat(lat,x_dist)),math.radians(270-yaw))
    temp = rotate((long,lat),(long,add_meters_to_lat(lat,x_dist)),math.radians(yaw))

    if x_pos == 0:
        return temp
    else:
        scalar = dist_to_focus * np.tan(np.arctan( 2 * abs(x_pos) * np.tan(math.radians(FOV/2)) / frame_width ))
        vector = scale_vector([[long,lat],[temp[0],temp[1]]],meters_to_lat(scalar))
        return vector[1]

def lat_long_point_in_range(frame_size,lat_long,x_min,x_max,y_min,y_max):
    '''
    Return point in frame based on lat long range
    '''
    range_width = x_max - x_min
    range_height = y_max - y_min
    x_point = (lat_long[0] - x_min) / range_width * frame_size[1]
    y_point = (y_max - lat_long[1]) / range_height * frame_size[0]
    return (int(x_point),int(y_point))

# Read flight log into dataframe
df = pd.read_csv('flight_record.csv',encoding="ISO-8859-1")

fly_state = df['OSD.flycState'].values
start_index = 0
for i, item in enumerate(fly_state):
    if item != 'EngineStart':
        start_index = i + 1
        break

# Flight time updates
flight_time = df['OSD.flyTime [s]'].values
flight_time = flight_time[start_index:] - flight_time[start_index]

# Video recording info
record_info = df['CAMERA_INFO.recordState'].values[start_index:]
record_info = np.where(record_info=='No',0,record_info)
record_info = np.where(record_info=='Starting',1,record_info)
video_delay_index = 0
for k, item in enumerate(record_info):
    if item == 1:
        video_delay_index = k
        break
video_delay_time = flight_time[video_delay_index]

# Find distance boundaries traveled in flight
gps_x_min = np.min(df['OSD.longitude'])
gps_x_max = np.max(df['OSD.longitude'])
gps_y_min = np.min(df['OSD.latitude'])
gps_y_max = np.max(df['OSD.latitude'])

# Drone positional information
x_pos = df['OSD.longitude'].values[start_index:]
y_pos = df['OSD.latitude'].values[start_index:]
alt = df['OSD.altitude [m]'].values[start_index:]
height = df['OSD.height [m]'].values[start_index:]
pitch = df['OSD.pitch'].values[start_index:]
yaw = df['OSD.yaw'].values[start_index:]
roll = df['OSD.roll'].values[start_index:]

# Gimbal positional information
gimbal_pitch = df['GIMBAL.pitch'].values[start_index:]
gimbal_roll = df['GIMBAL.roll'].values[start_index:]
gimbal_yaw = df['GIMBAL.yaw'].values[start_index:]

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

# Background image for plotting drone position and detections
background_img = np.ones((512,720,3), np.uint8) * 255

time = 0

# Adjust starting frame based on given start time
frame_number = np.max([0,int((time - video_delay_time) * fps)])
video.set(1, frame_number-1)

flight_point = lat_long_point_in_range(background_img.shape,(f_x_pos(time),f_y_pos(time)),gps_x_min,gps_x_max,gps_y_min,gps_y_max)
flight_points = [[flight_point[0]],[flight_point[1]]]
i = 1
while True:
    flight_point = lat_long_point_in_range(background_img.shape,(f_x_pos(time),f_y_pos(time)),gps_x_min,gps_x_max,gps_y_min,gps_y_max)
    flight_points[0].append(flight_point[0])
    flight_points[1].append(flight_point[1])
    current_height = np.round(f_height(time),2)
    object_map_points = []
    if f_record(time) == 1: # If camera is recording, show frame
        success, frame = video.read()
        if not success:
            break
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == 27:
            break

    if g_pitch(time) < 0:
        view_pitch = 90 + g_pitch(time)

        drone_focus_point = hover_point_to_lat_long(f_x_pos(time),f_y_pos(time),
                                                    current_height,-g_yaw(time),view_pitch,0)

        '''Insert NN for detection of objects in frame'''
        detections = [(0,0),(frame_width,0),(frame_width,frame_height),(0,frame_height)]

        detection_points = []
        for j, detection in enumerate(detections):
            pitch_rotation = (detection[1] - frame_center[1]) * degree_per_height
            yaw_rotation = (detection[0] - frame_center[0]) * degree_per_width
            total_pitch = view_pitch + pitch_rotation
            total_yaw = -g_yaw(time) + yaw_rotation
            if total_pitch < 0:
                total_yaw -= 2*yaw_rotation
            point = hover_point_to_lat_long(f_x_pos(time),f_y_pos(time),
                    current_height,total_yaw,total_pitch,detection[0]-frame_width/2)

            detection_points.append(lat_long_point_in_range(background_img.shape,(point[0],point[1]),gps_x_min,gps_x_max,gps_y_min,gps_y_max))

        background_img = np.ones((512,720,3), np.uint8) * 255
        for x in range(1,len(flight_points[0])):
            cv2.line(background_img,(flight_points[0][x-1],flight_points[1][x-1]),(flight_points[0][x],flight_points[1][x]),(255,0,0),2)
        for x in range(len(detection_points)):
            cv2.line(background_img,detection_points[x-1],detection_points[x],(0,255,0),2)
        draw_point = lat_long_point_in_range(background_img.shape,(drone_focus_point[0],drone_focus_point[1]),gps_x_min,gps_x_max,gps_y_min,gps_y_max)
        cv2.circle(background_img,draw_point,3,(0,0,255),2)
        cv2.imshow('plot',background_img)
        if cv2.waitKey(1) == 27:
            break

    else:
        cv2.line(background_img,(flight_points[0][-1],flight_points[1][-1]),(flight_points[0][-2],flight_points[1][-2]),(255,0,0),2)
        cv2.imshow('plot',background_img)


    time += 1/fps
    # time += 1.1
    i += 1.2
cv2.destroyAllWindows()



#
