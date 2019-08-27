
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import math
from geojson import Point
import geopy.distance

def find_pitch_and_yaw_adj(detection,current_pitch,current_yaw):
    current_s_height = np.sin(math.radians(current_pitch))

    x_hat = (detection[1] / frame_height) * s_height * np.sin(math.radians(current_pitch))
    y_hat = (detection[1] / frame_height) * s_height * np.cos(math.radians(current_pitch))

    new_x = np.cos(math.radians(current_pitch)) - x_hat
    new_y = s_width * (detection[0] / frame_width)
    new_z = current_s_height + y_hat

    pitch = np.arccos(new_z / np.sqrt(new_x**2 + new_y**2 + new_z**2))
    yaw = np.arctan(new_y / new_x)
    if new_x < 0:
        pitch = -pitch
        return -round(270 + math.degrees(pitch),2), current_yaw + round(math.degrees(yaw),2)
    else:
        return -round(math.degrees(pitch) - 90,2), current_yaw + round(math.degrees(yaw),2)

def pos_from_pitch_and_yaw(current_pos,current_height,pitch,yaw):
    mult = 1
    if pitch < -90:
        pitch = -(round(pitch + 180,2))
        mult = -1
    ground_dist = current_height / np.tan(math.radians(abs(pitch)))
    x_dist = mult * meters_to_long(current_pos[1],ground_dist * np.sin(math.radians(yaw)))
    y_dist = mult * meters_to_lat(ground_dist * np.cos(math.radians(yaw)))
    return (x_dist+current_pos[0],y_dist+current_pos[1])

def dist_between_lat_long_points(p1,p2):
    return geopy.distance.vincenty(p1, p2).m

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

height_width_ratio = frame_height/frame_width

s_width = np.tan(math.radians(FOV / 2))
s_height = np.tan(math.radians(height_width_ratio * FOV / 2))

# Background image for plotting drone position and detections
background_img = np.ones((512,720,3), np.uint8) * 255

time = 450

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

    view_pitch = g_pitch(time)
    if view_pitch < 0:

        new_pitch, new_yaw = find_pitch_and_yaw_adj((0,0),view_pitch,g_yaw(time))
        if new_pitch > 0:
            continue
        drone_focus_point = pos_from_pitch_and_yaw((f_x_pos(time),f_y_pos(time)),current_height,new_pitch,new_yaw)

        '''Insert NN for detection of objects in frame'''
        detections = [(-frame_width,-frame_height),(-frame_width,frame_height),(frame_width,frame_height),(frame_width,-frame_height)]

        detection_points = []
        for j, detection in enumerate(detections):

            new_pitch, new_yaw = find_pitch_and_yaw_adj(detection,view_pitch,g_yaw(time))
            if new_pitch > 0:
                new_pitch = -1
            point = pos_from_pitch_and_yaw((f_x_pos(time),f_y_pos(time)),current_height,new_pitch,new_yaw)

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
