import csv
import cv2
import copy
import glob
import math
from mpmath import mp
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import sys
import shutil
import argparse

from camera_pose_visualisation import *

unit_converstion_to_m = 1000

def lat_lng_to_selenographic_coordiantes(lat_rad, lng_rad, polar_radius):
    X = polar_radius*math.cos(lat_rad)*math.cos(lng_rad)
    Y = polar_radius*math.cos(lat_rad)*math.sin(lng_rad)
    Z = polar_radius*math.sin(lat_rad)
    return X, Y, Z

# Get a conic matrix from an ellipse.
def ellipse_to_conic_matrix(x, y, a, b, phi):
    A = a**2*((math.sin(phi))**2)+b**2*((math.cos(phi))**2)
    B = 2*(b**2-a**2)*math.cos(phi)*math.sin(phi)
    C = a**2*((math.cos(phi))**2)+b**2*((math.sin(phi))**2)
    D = -2*A*x-B*y
    E = -B*x-2*C*y
    F = A*x**2+B*x*y+C*y**2-a**2*b**2

    # TODO: do i need to normalise here?

    return np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

class CraterSimulated:

    # Initialise from simulated data.
    def __init__(self, X, Y, Z, a, b, phi):
        self.X, self.Y, self.Z, self.a, self.b, self.phi, self.radius = X, Y, Z, a, b, phi, a
        A = self.a**2*((math.sin(self.phi))**2)+self.b**2*((math.cos(self.phi))**2)
        B = 2*(self.b**2-self.a**2)*math.cos(self.phi)*math.sin(self.phi)
        C = self.a**2*((math.cos(self.phi))**2)+self.b**2*((math.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-self.a**2*self.b**2

        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

        self.id = "0-0"

    # Crater centre is on the plane of the crater rim.
    def get_crater_centre(self):
        return np.array([self.X, self.Y, self.Z])
    
    def get_crater_centre_hom(self):
        return np.array([self.X, self.Y, self.Z, 1])
    
    def project_crater_centre(self, Pm_c):
        X = np.dot(Pm_c, self.get_crater_centre_hom())
        return np.array([X[0]/X[2], X[1]/X[2]])
    
    def get_ENU(self):
        # Pc_M = self.get_crater_centre()
        # k = np.array([0, 0, 1])
        # u = Pc_M/np.linalg.norm(Pc_M)
        # e = np.cross(k, u)/np.linalg.norm(np.cross(k, u))
        # n = np.cross(u, e)/np.linalg.norm(np.cross(u, e))
        u = np.array([0, 0, 1])
        e = np.array([1, 0, 0])
        n = np.array([0, 1, 0])
        TE_M = np.transpose(np.array([e, n, u]))
        return TE_M
    
class CraterSimulatedFlipped:

    # Initialise from simulated data.
    def __init__(self, X, Y, Z, a, b, phi):
        self.X, self.Y, self.Z, self.a, self.b, self.phi, self.radius = X, Y, Z, a, b, phi, a
        A = self.a**2*((math.sin(self.phi))**2)+self.b**2*((math.cos(self.phi))**2)
        B = 2*(self.b**2-self.a**2)*math.cos(self.phi)*math.sin(self.phi)
        C = self.a**2*((math.cos(self.phi))**2)+self.b**2*((math.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-self.a**2*self.b**2

        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])

        self.id = "0-0"

    # Crater centre is on the plane of the crater rim.
    def get_crater_centre(self):
        return np.array([self.X, self.Y, self.Z])
    
    def get_crater_centre_hom(self):
        return np.array([self.X, self.Y, self.Z, 1])
    
    def project_crater_centre(self, Pm_c):
        X = np.dot(Pm_c, self.get_crater_centre_hom())
        return np.array([X[0]/X[2], X[1]/X[2]])
    
    def get_ENU(self):
        # n = np.array([0, math.cos((180)*math.pi/180), math.sin((180)*math.pi/180)])
        # e = np.array([1, 0, 0])
        # u = np.array([0, math.cos((90)*math.pi/180), math.sin((90)*math.pi/180)])
        u = np.array([0, math.cos((180)*math.pi/180), math.sin((180)*math.pi/180)])
        e = np.array([1, 0, 0])
        n = np.array([0, math.cos((90)*math.pi/180), math.sin((90)*math.pi/180)])
        TE_M = np.transpose(np.array([e, n, u]))
        return TE_M


class Crater:

    # Initialise craters from catalogue.
    def __init__(self, crater_dict, polar_radius, deg=True):
        self.crater_dict = crater_dict

        self.id = self.crater_dict["CRATER_ID"]

        # Check to see if the crater dictionary has additional stats.
        self.has_stats = False
        self.rim_height = 0
        self.rim_median = 0
        self.rim_mean = 0
        self.full_median = 0
        self.full_mean = 0
        if "rimMean" in self.crater_dict.keys():
            self.rim_height = self.crater_dict["rimMean"]
            self.rim_median = self.crater_dict["rimMedian"]
            self.rim_mean = self.crater_dict["rimMean"]
            self.full_median = self.crater_dict["fullMedian"]
            self.full_mean = self.crater_dict["fullMean"]
            self.has_stats = True

        # Initialise selenographic coordinates.
        lat = self.crater_dict["LAT_CIRC_IMG"]
        lng = self.crater_dict["LON_CIRC_IMG"]
        if deg:
            lat *= math.pi/180
            lng *= math.pi/180
        # Adjust the projected Z height.
        if self.has_stats:
            diff = ((self.rim_median))*unit_converstion_to_m
            # ((self.rim_median - self.full_median))*unit_converstion_to_m
            # print("self.rim_median",self.rim_median)
            # print("self.full_median", self.full_median)
            # print("self.rim_median-self.full_median",self.rim_median-self.full_median)
            # print("self.rim_median-self.full_median*1000",self.rim_median-self.full_median*1000)
            # print("polar_radius",polar_radius)
            # print("polar_radius+diff",polar_radius+diff)
            # print()
            self.X, self.Y, self.Z = lat_lng_to_selenographic_coordiantes(lat, lng, polar_radius+diff)
        else:
            self.X, self.Y, self.Z = lat_lng_to_selenographic_coordiantes(lat, lng, polar_radius)

        # Note: treating craters as circles
        self.diam = unit_converstion_to_m*self.crater_dict["DIAM_CIRC_IMG"]
        self.a = unit_converstion_to_m*self.crater_dict["DIAM_ELLI_MAJOR_IMG"]/2
        self.b = unit_converstion_to_m*self.crater_dict["DIAM_ELLI_MINOR_IMG"]/2
        # TODO: remove following line if you want to treat as ellipse.
        self.b = self.a
        self.phi = 0 # self.crater_dict["DIAM_ELLI_ANGLE_IMG"]*math.pi/180
        self.radius = self.a

        # Get the crater rim completeness.
        self.rim_completeness = self.crater_dict["ARC_IMG"]

        A = self.a**2*((math.sin(self.phi))**2)+self.b**2*((math.cos(self.phi))**2)
        B = 2*(self.b**2-self.a**2)*math.cos(self.phi)*math.sin(self.phi)
        C = self.a**2*((math.cos(self.phi))**2)+self.b**2*((math.sin(self.phi))**2)
        D = -2*A*0-B*(0)
        E = -B*0-2*C*(0)
        F = A*0**2+B*0*(0)+C*(0)**2-self.a**2*self.b**2

        self.conic_matrix_local = np.array([[A, B/2, D/2],[B/2, C, E/2],[D/2, E/2, F]])
    
    # Crater centre is on the plane of the crater rim.
    def get_crater_centre(self):
        return np.array([self.X, self.Y, self.Z])
    
    def get_crater_centre_hom(self):
        return np.array([self.X, self.Y, self.Z, 1])
    
    def project_crater_centre(self, Pm_c):
        X = np.dot(Pm_c, self.get_crater_centre_hom())
        return np.array([X[0]/X[2], X[1]/X[2]])
    
    def get_ENU(self):
        Pc_M = self.get_crater_centre()
        k = np.array([0, 0, 1])
        u = Pc_M/np.linalg.norm(Pc_M)
        e = np.cross(k, u)/np.linalg.norm(np.cross(k, u))
        n = np.cross(u, e)/np.linalg.norm(np.cross(u, e))
        TE_M = np.transpose(np.array([e, n, u]))
        return TE_M

# Get all png files.
def get_image_files(dir):
    image_files = glob.glob(dir+"*.png")
    image_files = [image_file[len(dir):] for image_file in image_files]
    return image_files

# Return poses of the camera from the PANGU flight file.
def get_ground_truth_poses(flight_file):
    f = open(flight_file, 'r')
    lines = f.readlines()
    lines = [i.split() for i in lines]
    poses = []
    for i in lines:
        # Camera pose line is prefixed with "start" and has structure: x, y, z, yaw, pitch, roll
        if len(i)>0 and i[0] == "start":
            pose = np.float_(i[1:])
            poses.append([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]])
    return poses

# Return the craters within a latiude and longitude range.
def filter_visable_craters(crater_list, lat_range, lng_range):
    min_lat = lat_range[0]
    max_lat = lat_range[1]
    westmost_lng = lng_range[0]
    eastmost_lng = lng_range[1]
    limited_craters = []
    for crater_obj in crater_list:
        crater = crater_obj.crater_dict
        try:
            if crater["LAT_CIRC_IMG"] > min_lat and crater["LAT_CIRC_IMG"] < max_lat and crater["LON_CIRC_IMG"] > westmost_lng and crater["LON_CIRC_IMG"] < eastmost_lng:
                limited_craters.append(crater_obj)
        except:
            print("ERROR: Crater database labels are in unexpected format.")
    return limited_craters

def get_LDEM_data(latitude_longitude_constraints_file):
    with open(latitude_longitude_constraints_file, mode ='r') as f:
        file = csv.reader(f)
        next(file) # Skip description line.
        max_latitude_deg, min_latitude_deg, westmost_longitude_deg,eastmost_longitude_deg, polar_radius_km = [float(x) for x in next(file)]
        
        # Set polar radius in metres.
        polar_radius = polar_radius_km*unit_converstion_to_m

        # Set lat/lng ranges.
        lat_range_deg = [min_latitude_deg, max_latitude_deg]
        lng_range_deg = [westmost_longitude_deg, eastmost_longitude_deg]

        return lat_range_deg, lng_range_deg, polar_radius

def get_craters(file, polar_radius):
    craters=[]
    with open(file, mode ='r')as file:
        csvFile = csv.reader(file)
        crater_identifiers = next(csvFile)
        for lines in csvFile:
            # TODO: check that this is right.
            if len(lines) == len(crater_identifiers):
                invalid_crater = False
                for i in range(len(lines)):
                    try:
                        lines[i] = float(lines[i])
                    except:
                        if lines[i] == '':
                            invalid_crater = True
                            break
                if not invalid_crater:
                    crater = Crater(dict(zip(crater_identifiers, lines)), polar_radius)
                    craters.append(crater)
    return craters

def get_camera_extrinsics(yaw_deg, pitch_deg, roll_deg, x, y, z, K):
    R_w_ci_intrinsic = R.from_euler('ZXZ',np.array([0,-90,0]),degrees=True).as_matrix()
    R_ci_cf_intrinsic = R.from_euler('ZXZ',np.array([yaw_deg, pitch_deg, 0]),degrees=True).as_matrix()
    R_c_intrinsic = np.dot(R_ci_cf_intrinsic, R_w_ci_intrinsic)
    R_w_c_extrinsic = np.linalg.inv(R_c_intrinsic)
    R_c_roll_extrinsic = R.from_euler('xyz',np.array([0,0,roll_deg]),degrees=True).as_matrix()
    R_w_c = np.dot(R_c_roll_extrinsic,R_w_c_extrinsic)
    Tm_c = R_w_c

    rm = np.array([x, y, z]) # position of camera in the moon reference frame
    rc = np.dot(Tm_c, -1*rm) # position of camera in the camera reference frame
    so3 = np.empty([3,4])
    so3[0:3, 0:3] = Tm_c
    so3[0:3,3] = rc 
    Pm_c = np.dot(K, so3)
    return Pm_c, Tm_c, rm

# Returns a camera intrinsic matrix.
def get_intrinsics(calibration_file):
    f = open(calibration_file, 'r')
    lines = f.readlines()
    calibration = lines[1].split(',')
    fov_deg = float(calibration[0])
    image_width = int(calibration[3])
    image_height = int(calibration[4])

    fov = fov_deg*math.pi/180
    fx = image_width/(2*math.tan(fov/2)) # Conversion from fov to focal legth
    fy = image_height/(2*math.tan(fov/2)) # Conversion from fov to focal legth
    cx = image_width/2
    cy = image_height/2

    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

# Make a point homogeneous.
def make_homogeneous(point):
    hom_point = np.ones(len(point)+1)
    hom_point[:len(point)] = point
    return hom_point

# Project a point.
def project_point(point, projection_matrix):
    hom_point = make_homogeneous(point)
    hom_projection = np.dot(projection_matrix, hom_point)
    camera_projection = [hom_projection[0]/hom_projection[2], hom_projection[1]/hom_projection[2]]
    return camera_projection

# Project crater centre coordinates onto image plane.
def project_crater_centre(crater, projection_matrix):
    return project_point(crater.get_crater_centre(), projection_matrix)

# Get projected crater conic.
def project_conics(c, Pm_c):
    k = np.array([0, 0, 1])
    Tl_m = c.get_ENU()#np.eye(3) # define a local coordinate system
    S = np.vstack((np.eye(2), np.array([0,0])))
    Pc_mi = c.get_crater_centre().reshape((3,1)) # get the real 3d crater point in moon coordinates
    Hmi = np.hstack((np.dot(Tl_m,S), Pc_mi))
    Cstar = np.linalg.inv(c.conic_matrix_local)
    Hci  = np.dot(Pm_c, np.vstack((Hmi, np.transpose(k))))
    Astar = np.dot(Hci,np.dot(Cstar, np.transpose(Hci)))
    A = np.linalg.inv(Astar)
    return A

# Get elliptical parameters from a conic matrix.
# Returns all 0's if invalid.
def conic_matrix_to_ellipse(cm):
    A = cm[0][0]
    B = cm[0][1]*2
    C = cm[1][1]
    D = cm[0][2]*2
    E = cm[1][2]*2
    F = cm[2][2]

    x_c = (2*C*D-B*E)/(B**2-4*A*C)
    y_c = (2*A*E-B*D)/(B**2-4*A*C)

    if ((B**2-4*A*C) >= 0):
        return 0,0,0,0,0

    try:
        a = math.sqrt((2*(A*E**2+C*D**2 - B*D*E + F*(B**2-4*A*C)))/((B**2-4*A*C)*(math.sqrt((A-C)**2+B**2)-A-C)))
        b = math.sqrt((2*(A*E**2+C*D**2 - B*D*E + F*(B**2-4*A*C)))/((B**2-4*A*C)*(-1*math.sqrt((A-C)**2+B**2)-A-C)))

        phi = 0
        if (B == 0 and A > C):
            phi = math.pi/2
        elif (B != 0 and A <= C):
            phi = 0.5*mp.acot((A-C)/B)
        elif (B != 0 and A > C):
            phi = math.pi/2+0.5*mp.acot((A-C)/B)
        
        return x_c, y_c, a, b, phi
    
    except:
        return 0,0,0,0,0




# Returns a cv2 image. 
def get_image(pangu_image_file):
    image = cv2.imread(pangu_image_file)
    return image

# Project a point onto the image plane.
def plot_point_on_image(point, image):
    center_coordinates = (int(point[0]), int(point[1]))
    axesLength = (1,1)
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (0, 255, 0)
    thickness = 1
    image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
    return image

def plot_conic_on_image_simulated(crater, Pm_c, image, f_projected_ellipses, f_projected_bounding_box, f_selenographic_craters, intensity = 1):
    x_c, y_c, a, b, phi = conic_matrix_to_ellipse(project_conics(crater, Pm_c))
    # print("Proj. rim:  ",round(x_c), round(y_c), (a,1), (b,1), (phi,1))
    x_p, y_p = crater.project_crater_centre(Pm_c)
    # print("Proj. centre:",round(x_p), round(y_p))
    # print("Diff:",np.linalg.norm(np.array([x_c, y_c]) - np.array([x_p, y_p])))
    # Check that an ellipse is valid.

    # Get bounding box.
    xa = np.sqrt((a**2*math.cos(phi)**2)+(b**2*math.sin(phi)**2))
    ya = np.sqrt((a**2*math.sin(phi)**2)+(b**2*math.cos(phi)**2))

    top_left = (round(-xa + x_c), round(-ya + y_c))
    bottom_right = (round(xa + x_c), round(ya + y_c))

    f_projected_ellipses.write(str(x_c)+", "+str(y_c)+", "+str(a)+", "+str(b)+", "+str(phi)+", "+str(crater.id)+"\n")
    f_projected_bounding_box.write(str(int(top_left[0]))+", "+str(int(top_left[1]))+", "+str(int(bottom_right[0]))+", "+str(int(bottom_right[1]))+"\n")
    f_selenographic_craters.write(str(crater.X)+", "+str(crater.Y)+", "+str(crater.Z)+", "+str(crater.a)+", "+str(crater.b)+", "+str(crater.phi)+", "+str(crater.id)+"\n")

    center_coordinates = (round(x_c), round(y_c))
    axesLength = (round(a), round(b))
    angle = round(phi*180/math.pi)
    startAngle = 0
    endAngle = 360
    color = (0, 255*intensity, 0)
    thickness = 1
    image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
    image = cv2.ellipse(image, center_coordinates, (1,1), angle, startAngle, endAngle, color, 2)
    image = cv2.ellipse(image, (round(x_p), round(y_p)), (1,1), 0, startAngle, endAngle, (0, 0, 255), 2)
    # image = cv2.ellipse(image, (256, 256), (1,1), 0, startAngle, endAngle, (0, 255, 255), 2)
    image = cv2.rectangle(image, top_left, bottom_right, (255,0,255), 1 )
    return image, x_c, y_c, a, b, phi, x_p, y_p

def plot_conic_on_image(crater, Pm_c, image, f_projected_ellipses, f_projected_bounding_box, f_selenographic_craters, intensity = 1):
    x_c, y_c, a, b, phi = conic_matrix_to_ellipse(project_conics(crater, Pm_c))
    x_p, y_p = crater.project_crater_centre(Pm_c)
    # Check that an ellipse is valid.

    # Get bounding box.
    xa = np.sqrt((a**2*math.cos(phi)**2)+(b**2*math.sin(phi)**2))
    ya = np.sqrt((a**2*math.sin(phi)**2)+(b**2*math.cos(phi)**2))

    top_left = (round(-xa + x_c), round(-ya + y_c))
    bottom_right = (round(xa + x_c), round(ya + y_c))

    f_projected_ellipses.write(str(x_c)+", "+str(y_c)+", "+str(a)+", "+str(b)+", "+str(phi)+", "+str(crater.id)+"\n")
    f_projected_bounding_box.write(str(int(top_left[0]))+", "+str(int(top_left[1]))+", "+str(int(bottom_right[0]))+", "+str(int(bottom_right[1]))+"\n")
    f_selenographic_craters.write(str(crater.X)+", "+str(crater.Y)+", "+str(crater.Z)+", "+str(crater.a)+", "+str(crater.b)+", "+str(crater.phi)+", "+str(crater.id)+"\n")

    center_coordinates = (int(x_c), int(y_c))
    axesLength = (int(a), int(b))
    angle = int(phi*180/math.pi)
    startAngle = 0
    endAngle = 360
    color = (0, 255*intensity, 0)
    thickness = 1
    image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
    image = cv2.ellipse(image, center_coordinates, (1,1), angle, startAngle, endAngle, color, 2)
    image = cv2.ellipse(image, (int(x_p), int(y_p)), (1,1), 0, startAngle, endAngle, (0, 0, 255), 2)
    image = cv2.rectangle(image, top_left, bottom_right, (255,0,255), 1 )
    return image

def crater_in_camera_view(crater, rm, projection_filtering):
    crater_normal = (crater.get_ENU()[:,2])/np.linalg.norm(crater.get_ENU()[:,2])
    camera_to_crater_v = (np.array([rm[0]-crater.X, rm[1]-crater.Y, rm[2]-crater.Z]))/np.linalg.norm(np.array([rm[0]-crater.X, rm[1]-crater.Y, rm[2]-crater.Z]))
    angle = np.arccos(min(np.dot(camera_to_crater_v, crater_normal) / (np.linalg.norm(camera_to_crater_v) * np.linalg.norm(crater_normal)), 0.99999999999))
    # Strictly within the observed surface, set - angle*180/np.pi <= 90 and angle*180/np.pi >= 0.
    # Following parameters indicated that some craters far away are unlikely to be detected by the camera.
    if (projection_filtering and angle*180/np.pi <= 75 and angle*180/np.pi >= 0):
        return True
    elif (not projection_filtering and angle*180/np.pi <= 90 and angle*180/np.pi >= 0):
        return True
    else:
        return False

def main():

    parser = argparse.ArgumentParser(description='Process files for crater projection.')
    # Expected directory and file structure.
    parser.add_argument('write_dir') 
    parser.add_argument('crater_catalogue_file')
    parser.add_argument('calibration_file')
    parser.add_argument('latitude_longitude_constraints_file')
    parser.add_argument('flight_file')
    parser.add_argument('images_dir')
    parser.add_argument('--projection_filtering',action='store_true')  # on/off flag - True: we filter craters based on the imaged projected craters
    parser.add_argument('--simulated',action='store_true')  # on/off flag - True: we filter craters based on the imaged projected craters
    args = parser.parse_args()

    write_dir = args.write_dir
    crater_catalogue_file = args.crater_catalogue_file
    calibration_file = args.calibration_file
    latitude_longitude_constraints_file = args.latitude_longitude_constraints_file
    flight_file = args.flight_file
    images_dir = args.images_dir
    projection_filtering = args.projection_filtering
    simulated = args.simulated

    # Minimum number of craters we expect in a scene.
    # TODO: set back to 3
    min_craters = 1

    # Writing directories.
    ground_truth_images_dir = "ground_truth_images/"
    ground_truth_ellipse_and_bounding_box_images_dir = "ground_truth_ellipse_and_bounding_box_images/"
    ground_truth_projected_ellipses_dir = "ground_truth_projected_ellipses/"
    ground_truth_bounding_boxes_dir = "ground_truth_bounding_boxes/"
    ground_truth_selenographic_crater_coordinates_dir = "ground_truth_selenographic_crater_coordinates/"

    if not os.path.isdir(write_dir+ground_truth_images_dir):
        os.makedirs(write_dir+ground_truth_images_dir)
    else:
        shutil.rmtree(write_dir+ground_truth_images_dir)
        os.makedirs(write_dir+ground_truth_images_dir)

    if not os.path.isdir(write_dir+ground_truth_ellipse_and_bounding_box_images_dir):
        os.makedirs(write_dir+ground_truth_ellipse_and_bounding_box_images_dir)
    else:
        shutil.rmtree(write_dir+ground_truth_ellipse_and_bounding_box_images_dir)
        os.makedirs(write_dir+ground_truth_ellipse_and_bounding_box_images_dir)
    
    if not os.path.isdir(write_dir+ground_truth_projected_ellipses_dir):
        os.makedirs(write_dir+ground_truth_projected_ellipses_dir)
    else:
        shutil.rmtree(write_dir+ground_truth_projected_ellipses_dir)
        os.makedirs(write_dir+ground_truth_projected_ellipses_dir)
    
    if not os.path.isdir(write_dir+ground_truth_bounding_boxes_dir):
        os.makedirs(write_dir+ground_truth_bounding_boxes_dir)
    else:
        shutil.rmtree(write_dir+ground_truth_bounding_boxes_dir)
        os.makedirs(write_dir+ground_truth_bounding_boxes_dir)
    
    if not os.path.isdir(write_dir+ground_truth_selenographic_crater_coordinates_dir):
        os.makedirs(write_dir+ground_truth_selenographic_crater_coordinates_dir)
    else:
        shutil.rmtree(write_dir+ground_truth_selenographic_crater_coordinates_dir)
        os.makedirs(write_dir+ground_truth_selenographic_crater_coordinates_dir)

        
    # Writing files.
    ground_flight_file = "ground_truth_flight.fli"
    crater_filtering_criteria_file = "crater_filtering_criteria.csv"
    error_bounds_file = "position_attitude_error_bounds.csv"

    # Get LDEM data.
    lat_range_deg, lng_range_deg, polar_radius = get_LDEM_data(latitude_longitude_constraints_file)

    # If it's siumated data, the craters won't be taken from the crater catalogue.
    if simulated:
        craters = [CraterSimulated(0, 0, 0, 400, 400, 0), CraterSimulatedFlipped(0, 0, -57.75, 408.15, 408.15, 0)]

    else:
        # Get list of craters.
        craters = get_craters(crater_catalogue_file, polar_radius)

        # Get craters within the lat/lng bounds.
        craters = filter_visable_craters(craters, lat_range_deg, lng_range_deg)

    # Get camera intrinsic matrix.
    K = get_intrinsics(calibration_file)

    # Get the camera extrinsic matrix from pose.
    ground_truth_poses = get_ground_truth_poses(flight_file)
    flight_file = open(write_dir+ground_flight_file, "w")

    # Get the image files.
    image_files = get_image_files(images_dir)
    image_files.sort(key=lambda x: int(x[:-4]))

    # Write the crater filtering criteria file.
    f_crater_filtering_criteria = open(write_dir+crater_filtering_criteria_file,"w")
    f_crater_filtering_criteria.write("ARC_IMG_threshold, diam_min, diam_max, semi_major_minor_axis_thresh\n")

    rim_completeness_threshold = 0.9
    diam_min = 4*unit_converstion_to_m #10
    diam_max = 125*unit_converstion_to_m #50
    semi_major_minor_axis_thresh = 1.1

    f_crater_filtering_criteria.write(str(rim_completeness_threshold)+", "+str(diam_min)+", "+str(diam_max)+", "+str(semi_major_minor_axis_thresh)+"\n")
    f_crater_filtering_criteria.close()

    # Write the error bound file.
    f_error_bounds = open(write_dir+error_bounds_file,"w")
    f_error_bounds.write("position_error_bound_m, attitude_error_bound_deg\n")
    position_error = 10000
    attitude_error = 0.1
    f_error_bounds.write(str(position_error)+","+str(attitude_error)+"\n")
    f_error_bounds.close()

    start = 0
    end = len(ground_truth_poses)
    step = 1

    all_proj_ellipses = []
    points = []
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(0,0,0,c="b")
    ax.scatter(0,0,0,c="r")
    t = np.arange(0, 10*np.pi, np.pi/50)
    x = 400*np.sin(t)
    y = 400*np.cos(t)

    ax.plot3D(x, y, 0, c = "r")
    ax.grid()
    ax.set_title('3D Scatter Plot')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    # print("REFLECTION")
    c = np.array([1/2, math.sqrt(3)/2])
    b = np.array([0, 1])
    c_p = c-2*(np.dot(c, b))*b
    # print(c_p, np.array([1/2, -math.sqrt(3)/2]))

    for j in range(start, end, step):
        x, y, z, yaw_deg, pitch_deg, roll_deg = ground_truth_poses[j]
        Pm_c, _, rm = get_camera_extrinsics(yaw_deg, pitch_deg, roll_deg, x, y, z, K)

        # Read image.
        file_name = image_files[j][:-4]
        image = get_image(images_dir+image_files[j])
        image_width, image_height, _ = image.shape

        sub_craters = copy.deepcopy(craters)

        if not simulated:

            #### Crater filtering #####
            rest_craters = []
            # Store only the craters which project onto the image plane.
            for i in range(len(sub_craters)):
                x_c, y_c, a, b, phi =  conic_matrix_to_ellipse(project_conics(sub_craters[i], Pm_c))
                if (x_c != 0 and y_c != 0 and a != 0 and b != 0 and phi != 0):
                    if x_c >= 0 and x_c < image_width and y_c >= 0 and y_c < image_height:
                        if x_c - a*math.cos(phi) >= 0 and x_c + a*math.cos(phi) < image_width:
                                if y_c - a*math.sin(phi) >= 0 and y_c + a*math.sin(phi) < image_height:
                                        rest_craters.append(sub_craters[i])
            sub_craters = rest_craters

            # Store craters whose plane normal is within the camera orientation.
            rest_craters = []
            for i in range(len(sub_craters)):
                if (crater_in_camera_view(sub_craters[i], rm, projection_filtering)):
                    rest_craters.append(sub_craters[i])
            sub_craters = rest_craters

            # Filter craters within Christian's parameters.
            # if (projection_filtering):
            rest_craters = []
            for i, crater in enumerate(sub_craters):
                if (crater.rim_completeness > rim_completeness_threshold and crater.diam >= diam_min and crater.diam <= diam_max and crater.a/crater.b <= semi_major_minor_axis_thresh):
                    rest_craters.append(sub_craters[i])
            sub_craters = rest_craters

            # Filter craters on fake CDA requirements.
            # Store only the craters that have a projected semi minor axis > 3 pixels.
            if (projection_filtering):
                rest_craters = []
                for i in range(len(sub_craters)):
                    _, _, a, b, _ =  conic_matrix_to_ellipse(project_conics(sub_craters[i], Pm_c))
                    # If the semi minor axis is big enough (e.g. 10 pixels), we can assume we'd detect it.
                    # If the semi minor is smaller (5 pixels) and the ellipticity is small enough (but not too small), we'd detect it.
                    if b >= 10 or (b >= 5 and b/a >= 0.75):
                        rest_craters.append(sub_craters[i])
                sub_craters = rest_craters

            # TODO: remove
            # sub_craters = [sub_craters[0]]


            # If the craters store additional stats ...
            if (projection_filtering):
                if (len(sub_craters) > 0 and sub_craters[0].has_stats):
                    # Store only the craters that have rim height greater than the rim threshold.
                    rest_craters = []
                    rim_threshold = 0.09 #km
                    for i in range(len(sub_craters)):
                        diff = ((sub_craters[i].rim_median - sub_craters[i].full_median)/sub_craters[i].b)*1000
                        if ( (diff > 0.04 and sub_craters[i].b > 5000) or (diff > rim_threshold)): #
                            rest_craters.append(sub_craters[i])
                    sub_craters = rest_craters


            crater_rim_completenesses = [crater.rim_completeness for crater in sub_craters]

        if len(sub_craters) >= min_craters:
            ###############

            ##### Image filtering #####
            # If the bound of the DEM are in the image, discard the image.
            max_portion_of_black_pixels = 0.6
            num_black_pixels = 0
            # TODO: uncomment if you want to filter out massivley blacked-out areas.
            for r in range(image_height):
                for c in range(image_width):
                    if image[r, c][0] == 0 and image[r, c][1] == 0 and image[r, c][2] == 0:
                        num_black_pixels += 1

            if num_black_pixels/(image_height*image_width) < max_portion_of_black_pixels or not projection_filtering:
                # If we don't have a minimum number of 3 craters that can be seen in the scene, we exclude this image.
                if len(sub_craters) >= min_craters:
                    print("     ",file_name)

                    # Save projection data.
                    f_projected_ellipses = open(write_dir+ground_truth_projected_ellipses_dir+file_name+".txt", "w")
                    f_projected_ellipses.write("ellipse: x_centre, y_centre, semi_major_axis, semi_minor_axis, rotation, id\n")
                    f_projected_bounding_box = open(write_dir+ground_truth_bounding_boxes_dir+file_name+".txt", "w")
                    f_projected_bounding_box.write("bounding box: top left(x, y), bottom right (x, y)\n")
                    f_selenographic_craters = open(write_dir+ground_truth_selenographic_crater_coordinates_dir+file_name+".txt", "w")
                    f_selenographic_craters.write("selenographic crater coordinates: X, Y, Z, a_metres, b_metres, id\n")

                    # Write line to flight file.
                    pose_str = "start " + str(x) +" "+ str(y) +" "+ str(z) +" "+ str(yaw_deg) +" "+ str(pitch_deg) +" "+ str(roll_deg) + "\n"
                    points.append([x, y, z])
                    
                    ax.scatter(x, y, z, c = 'r')
                    flight_file.write(pose_str)
                    # print(np.linalg.norm(np.array([x, y, z])))

                    # Save image.
                    cv2.imwrite(write_dir+ground_truth_images_dir+file_name+".png", image)

                    # Plot the crater ellipses onto the image plane.
                    # Filter the craters.
                    for i, crater in enumerate(sub_craters):
                        # r = max(0.00000001, max(crater_rim_completenesses)-min(crater_rim_completenesses))
                        if not simulated:
                            image = plot_conic_on_image(crater, Pm_c, image, f_projected_ellipses, f_projected_bounding_box, f_selenographic_craters)#, intensity=(crater.rim_completeness-min(crater_rim_completenesses))/r)
                        else:
                            image, x_c, y_c, a, b, phi, x_p, y_p = plot_conic_on_image_simulated(crater, Pm_c, image, f_projected_ellipses, f_projected_bounding_box, f_selenographic_craters)#, intensity=(crater.rim_completeness-min(crater_rim_completenesses))/r)
                            all_proj_ellipses.append([x_c, y_c, a, b, phi, x_p, y_p])
                    cv2.imwrite(write_dir+ground_truth_ellipse_and_bounding_box_images_dir+file_name+".png", image)
                    # cv2.imshow("name", image)

                    f_projected_ellipses.close()
                    f_projected_bounding_box.close()
                    f_selenographic_craters.close()
                else:
                    print("num craters - discarded",file_name)
            else:
                    print("colour - discarded",file_name)
        else:
            print("num craters - discarded",file_name)
    
    if simulated:

        # plt.show()

        image = np.zeros((512,512,3), np.uint8)
        image = cv2.imwrite(write_dir+ground_truth_ellipse_and_bounding_box_images_dir+"overlapped_ellipses.png", image)
        image = cv2.imread(write_dir+ground_truth_ellipse_and_bounding_box_images_dir+"overlapped_ellipses.png")
        for proj_ellipse in all_proj_ellipses:
            x_c, y_c, a, b, phi, x_p, y_p = proj_ellipse
            image = cv2.ellipse(image, (round(x_c), round(y_c)), (round(a), round(b)), round(phi*180/math.pi), 0, 360, (0, 255, 0), 2)
            image = cv2.ellipse(image, (round(x_c), round(y_c)), (1,1), 0, 0, 360, (0, 255, 0), 2)
            image = cv2.ellipse(image, (round(x_p), round(y_p)), (1,1), 0, 0, 360, (0, 0, 255), 2)
            # image = cv2.ellipse(image, (256, 256), (1,1), 0, 0, 360, (0, 255, 255), 2)
        image = cv2.imwrite(write_dir+ground_truth_ellipse_and_bounding_box_images_dir+"overlapped_ellipses.png", image)
    flight_file.close()

if __name__ == "__main__":
    main()
