
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
import math
from geojson import Point
import geopy.distance

def free_norm(vector):
    temp = np.array([[0,0],[vector[0,1]-vector[0,0],vector[1,1]-vector[1,0]]])
    return np.linalg.norm(temp)

def dist_between_lat_long_points(p1,p2):
    return geopy.distance.vincenty(p1, p2).m

def scale_vector(origin, point, scalar):
    '''Scale a vector that isn't through origin'''
    temp = np.array([[0,0],[point[0]-origin[0],point[1]-origin[1]]])
    scaled = scalar * temp / free_norm(temp)
    scaled[:,0] += origin[0]
    scaled[:,1] += origin[1]
    return scaled

def rotate(origin, point, angle):
    '''Rotate a vector that isn't thorugh origin'''
    new_x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    new_y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    return (new_x, new_y)

earth_radius = 6371000.0
def add_meters_to_lat(lat,dist):
    return lat  + (dist / earth_radius) * (180 / np.pi)
def add_meters_to_long(lat,long,dist):
    return long + (dist / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi/180)
def meters_to_lat(dist):
    return (dist / earth_radius) * (180 / np.pi)
def meters_to_long(lat,dist):
    return (dist / earth_radius) * (180 / np.pi) / np.cos(lat * np.pi/180)


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

# Get frames per second info from video
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver) < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video.get(cv2.CAP_PROP_FPS)

# Create interactive plot and limit to flight area
plt.xlim(gps_x_min,gps_x_max)
plt.ylim(gps_y_min,gps_y_max)
plt.ion()
plt.show()

time = 0
flight_points = [[f_x_pos(time)],[f_y_pos(time)]]
i = 1
while True:
    object_map_points = []
    if f_record(time) == 1: # If camera is recording, show frame
        success, frame = video.read()
        # if not success:
        #     break
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) == 27:
        #     break

    # time += 1/fps
    time += 1
    flight_points[0].append(float(f_x_pos(time)))
    flight_points[1].append(float(f_y_pos(time)))

    current_height = np.round(f_height(time),2)
    # print('Pitch: {}, Yaw: {}, Dist: {}'.format(np.round(g_pitch(time),2),np.round(g_yaw(time),2),current_height))
    if g_pitch(time) < -FOV/2:
        theta = 90-g_pitch(time)
        dist_to_focus = current_height/np.cos(90-theta)
        x_dist = np.tan(theta)*current_height
        drone_focus_point = rotate((flight_points[0][-1],flight_points[1][-1]),(flight_points[0][-1],add_meters_to_lat(flight_points[1][-1],x_dist)),math.radians(180-g_yaw(time)))

        ground_vector = np.array([[flight_points[0][-1],flight_points[1][-1]],[drone_focus_point[0],drone_focus_point[1]]])

        # Find transformation
        pts_src = np.array([[int(frame.shape[1]/2), int(frame.shape[0]/2)], # Middle
                            [0, int(frame.shape[0]/2)], # Middle Left
                            [frame.shape[1],int(frame.shape[0]/2)], # Middle Right
                            [int(frame.shape[1]/2), 0], # Top Middle
                            [int(frame.shape[1]/2), frame.shape[0]]]) # Bottom Middle
        horizontal_view_dist = np.tan(FOV/2) * current_height / np.cos(theta)
        vertical_view_dist_up = np.sin(FOV/2) * dist_to_focus / np.sin(90 - theta - FOV/2)
        vertical_view_dist_down = np.sin(FOV/2) * dist_to_focus / np.sin(90 - theta)

        view_extension_up = vertical_view_dist_up * np.tan(90 - theta)
        view_extension_down = vertical_view_dist_down * np.tan(90 - theta)

        pts_dst = np.array([[drone_focus_point[0], drone_focus_point[1]],
                            [489, drone_focus_point[1]],
                            [505, drone_focus_point[1]],
                            [drone_focus_point[0], 235],
                            [drone_focus_point[0],153]])
        h, status = cv2.findHomography(pts_src, pts_dst)
        # Detection flight_points
        a = np.array([[int(frame.shape[1]/2), int(frame.shape[0]/2)]], dtype='float32')
        a = np.array([a])
        # Map flight_points
        map_points = cv2.perspectiveTransform(a, h)

        plt.plot(flight_points[0][-2:],flight_points[1][-2:],'blue')
        for point in map_points:
            object_map_points.append(plt.scatter(point[0][0],point[0][1],c='orange'))

        rotate_point1 = rotate(ground_vector[1],ground_vector[0],math.radians(90))
        rotate_point2 = rotate(ground_vector[1],ground_vector[0],math.radians(270))
        rotate_point3 = rotate(ground_vector[1],ground_vector[0],math.radians(0))
        rotate_point4 = rotate(ground_vector[1],ground_vector[0],math.radians(180))

        left_vector = np.array([[ground_vector[1][0],rotate_point1[0]],[ground_vector[1][1],rotate_point1[1]]])
        right_vector = np.array([[ground_vector[1][0],rotate_point2[0]],[ground_vector[1][1],rotate_point2[1]]])
        down_vector = np.array([[ground_vector[1][0],rotate_point3[0]],[ground_vector[1][1],rotate_point3[1]]])
        up_vector = np.array([[ground_vector[1][0],rotate_point4[0]],[ground_vector[1][1],rotate_point4[1]]])

        corner_point1 = rotate(up_vector[:,1],up_vector[:,0],math.radians(270))
        corner_vector1 = np.array([[up_vector[0][1],corner_point1[0]],[up_vector[1][1],corner_point1[1]]])
        normed_extension_up = free_norm(corner_vector1) * view_extension_up / dist_between_lat_long_points(corner_vector1[:,0],corner_vector1[:,1])
        top_left_vector = scale_vector(corner_vector1[:,0],corner_vector1[:,1],free_norm(corner_vector1)+normed_extension_up)

        corner_point2 = rotate(down_vector[:,1],down_vector[:,0],math.radians(90))
        corner_vector2 = np.array([[down_vector[0][1],corner_point2[0]],[down_vector[1][1],corner_point2[1]]])
        normed_extension_down = free_norm(corner_vector2) * view_extension_down / dist_between_lat_long_points(corner_vector2[:,0],corner_vector2[:,1])
        bottom_left_vector = scale_vector(corner_vector2[:,0],corner_vector2[:,1],free_norm(corner_vector2)+normed_extension_down)

        c = plt.plot(left_vector[0],left_vector[1],'blue')
        d = plt.plot(right_vector[0],right_vector[1],'blue')
        e = plt.plot(up_vector[0],up_vector[1],'blue')
        f = plt.plot(down_vector[0],down_vector[1],'blue')
        c2 = plt.plot(top_left_vector[:,0],top_left_vector[:,1],'red')
        d2 = plt.plot(bottom_left_vector[:,0],bottom_left_vector[:,1],'red')

        plt.pause(.001)
        for point in object_map_points:
            point.remove()
        c[0].remove()
        d[0].remove()
        e[0].remove()
        f[0].remove()
        c2[0].remove()
        d2[0].remove()

    else:
        plt.plot(flight_points[0][-2:],flight_points[1][-2:],'blue')
        plt.pause(.001)


    i += 1
# cv2.destroyAllWindows()





#
