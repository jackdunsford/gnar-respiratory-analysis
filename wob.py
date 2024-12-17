import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from collections import OrderedDict
import spirometry
from shapely.geometry import Polygon
import scipy.integrate
from shapely.geometry import LineString, Point, MultiPoint
import os 

def plot_triangle_and_calculate_area(p1, p2, p3, col, edge):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the area using the determinant formula
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    # Plot the triangle
    triangle_x = [x1, x2, x3, x1]
    triangle_y = [y1, y2, y3, y1]

    # plt.scatter([x1, x2, x3], [y1, y2, y3], color="red", zorder=5)
    plt.fill(triangle_x[:-1], triangle_y[:-1], alpha=0.6, label="Filled Triangle", color=col, edgecolor=edge)
    return area

def get_points(averageexpbreath, age, sex, frc_y):
    y_eelv = averageexpbreath['volume'].iloc[0]
    y_eilv = averageexpbreath['volume'].iloc[-1]
    x_eelv = averageexpbreath['poes'].iloc[0]
    x_eilv = averageexpbreath['poes'].iloc[-1]

    if sex == 1:
        if age < 40:
            ccw_slope = 0.204
        elif age > 40 < 55:
            ccw_slope = 0.174
        else: ccw_slope = 0.149
    else:
        if age < 40:
            ccw_slope = 0.187
        elif age > 40 < 55:
            ccw_slope = 0.153
        else: ccw_slope = 0.120

    # ccw_slope = 1/ccw_slope

    # Given data
    line1_start = (x_eilv, y_eilv)  # (x, y) of start point of Line 1
    line1_end = (x_eelv, y_eelv)        # (x, y) of end point of Line 1
    y_intersection = frc_y              # y-coordinate of the intersection
    slope_line2 = ccw_slope                  # Slope of Line 2
    y2_start = y_eilv               # Start y-coordinate of Line 2
    y2_end = y_eelv                     # End y-coordinate of Line 2

    # Equation of Line 1: y = m1 * x + b1
    m1 = (line1_end[1] - line1_start[1]) / (line1_end[0] - line1_start[0])
    b1 = line1_start[1] - m1 * line1_start[0]

    # Equation of Line 2: y = m2 * x + b2
    b2 = y_intersection - slope_line2 * (line1_start[0] + (y_intersection - b1) / m1)  # Use intersection logic

    # Calculate x-coordinate of intersection
    frc_x = (y_intersection - b1) / m1

    x_ccw_eilv = (y_eilv - frc_y) / slope_line2 + frc_x
    x_ccw_eelv = (y_eelv - frc_y) / slope_line2 + frc_x

    return x_eelv, x_eilv, y_eelv, y_eilv, frc_x, x_ccw_eilv, x_ccw_eelv

# Function to calculate intersection of two line segments
def line_intersection(p1, p2, q1, q2):
    """
    Find the intersection point of two line segments (p1 -> p2) and (q1 -> q2).
    Returns the intersection point (x, y) or None if no intersection.
    """
    # Line 1: p1 to p2
    x1, y1 = p1
    x2, y2 = p2
    
    # Line 2: q1 to q2
    x3, y3 = q1
    x4, y4 = q2

    # Determinant calculation
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Lines are parallel or coincident

    # Calculate intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Check if the intersection point is within both segments
    if (
        min(x1, x2) <= px <= max(x1, x2) and
        min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and
        min(y3, y4) <= py <= max(y3, y4)
    ):
        return (px, py)
    return None

def find_intersection(line_start, line_end, curve_points):
    # Create the line and curve as Shapely objects
    line = LineString([line_start, line_end])
    curve = LineString(curve_points)

    # Find the intersection between the line and the curve
    intersection = line.intersection(curve)

    return intersection

def exp_resistive(averageexpbreath, point_b, point_c, point_d):
    exp_curve = np.array(list(zip(averageexpbreath['poes'], averageexpbreath['volume'])))
    line_points = [point_d, point_c]
    intersection = find_intersection(point_c, point_d, exp_curve)
    if isinstance(intersection, MultiPoint):
        intersections = [p for p in intersection.geoms][0]
    else: intersections = intersection

    exp_res_tri = plot_triangle_and_calculate_area((intersections.x, intersections.y), (point_c), (point_b), 'blue', 'none')
    
    y_curve = averageexpbreath['volume'][averageexpbreath['volume'] <= intersections.y]
    x_curve = averageexpbreath['poes'].iloc[:len(y_curve)]
    x_line = np.linspace(point_b[0], intersections.x, len(x_curve))
    exp_res_curve = plt.fill_betweenx(y_curve, x_curve, x_line,color='blue',alpha=0.6, edgecolor='none')
    exp_curve_area = 0
    for path in exp_res_curve.get_paths():
        vertices = path.vertices
        exp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))

    return exp_res_tri + exp_curve_area

def insp_resistive(averageinspbreath, point_a,point_b, point_c, point_e):
    insp_curve = np.array(list(zip(averageinspbreath['poes'], averageinspbreath['volume'])))
    line_points = [point_e, point_c]

    intersection = find_intersection(point_e, point_c, insp_curve)
    if isinstance(intersection, MultiPoint):
        intersections = [p for p in intersection.geoms][0]
    else: intersections = intersection
    if intersections.is_empty:
        insp_res_tri = plot_triangle_and_calculate_area((point_b), (point_c), (point_a), 'red', 'none')
        y_curve = averageinspbreath['volume']
        x_curve = averageinspbreath['poes']
        x_line = np.linspace(point_b[0], point_a[0], len(x_curve))
        insp_res_curve = plt.fill_betweenx(y_curve, x_curve, x_line,color='red',alpha=0.6, edgecolor='none')
        insp_curve_area = 0
        for path in insp_res_curve.get_paths():
            vertices = path.vertices
            insp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                        np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    else:
        insp_res_tri = plot_triangle_and_calculate_area((intersections.x, intersections.y), (point_c), (point_a), 'red', 'none')
        y_curve = averageinspbreath['volume'][averageinspbreath['volume'] >= intersections.y]
        x_curve = averageinspbreath['poes'].iloc[len(averageinspbreath['poes'])-len(y_curve):]
        x_line = np.linspace(intersections.x,point_a[0], len(x_curve))
        insp_res_curve = plt.fill_betweenx(y_curve, x_curve, x_line,color='red',alpha=0.6, edgecolor='none')
        insp_curve_area = 0
        for path in insp_res_curve.get_paths():
            vertices = path.vertices
            insp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                        np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    return insp_res_tri + insp_curve_area

def hedstrand(averageexpbreath, averageinspbreath, point_a, point_b, ex_stage, pdf, settings):

    plt.figure(figsize=(5,7))
    plt.title("Hedstrand Diagram - " + ex_stage + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(averageexpbreath['poes'], averageexpbreath['volume'], color='black')
    plt.plot(averageinspbreath['poes'], averageinspbreath['volume'],color='black')
    insp_elastic = plot_triangle_and_calculate_area(point_a, point_b, (point_b[0], point_a[1]), 'green', 'black')

    exp_curve = np.array(list(zip(averageexpbreath['poes'][1:], averageexpbreath['volume'][1:])))
    line_points = [point_b, (point_b[0], point_a[1])]
    # intersections = []

    intersection = find_intersection(point_b, (point_b[0], point_a[1]), exp_curve)
    if isinstance(intersection, MultiPoint):
        intersections = [p for p in intersection.geoms][0]
    else: intersections = intersection


    exp_y_curve = averageexpbreath['volume'][averageexpbreath['volume'] <= intersections.y]
    exp_x_curve = averageexpbreath['poes'].iloc[:len(exp_y_curve)]
    exp_x_line = np.linspace(point_b[0], intersections.x, len(exp_x_curve))
    exp_res_curve = plt.fill_betweenx(exp_y_curve, exp_x_curve, exp_x_line,color='blue',alpha=0.6, edgecolor='none')
    exp_curve_area = 0
    for path in exp_res_curve.get_paths():
        vertices = path.vertices
        exp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))

    
    insp_y_curve = averageinspbreath['volume']
    insp_x_curve = averageinspbreath['poes']
    insp_x_line = np.linspace(point_b[0], point_a[0], len(insp_x_curve))
    insp_res_curve = plt.fill_betweenx(insp_y_curve, insp_x_curve, insp_x_line,color='red',alpha=0.6, edgecolor='none')
    insp_curve_area = 0
    for path in insp_res_curve.get_paths():
        vertices = path.vertices
        insp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))

    return insp_curve_area, insp_elastic, exp_curve_area, 0

def modified_cambell(averageexpbreath, averageinspbreath, frc, ex_stage, pdf, settings):
    x_eelv, x_eilv, y_eelv, y_eilv, frc_x, x_ccw_eilv, x_ccw_eelv = get_points(averageexpbreath, 25, 1, frc)
    point_a = (x_eilv, y_eilv) #end insp
    point_b = (x_eelv, y_eelv) #end exp
    point_c = (frc_x, frc) #frc
    point_d = (x_ccw_eilv, y_eilv) #ccw end insp
    point_e = (x_ccw_eelv, y_eelv) #ccw end exp
    
    plt.figure(figsize=(5,7))
    plt.title("Modified Campbell diagram - " + ex_stage + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(averageexpbreath['poes'], averageexpbreath['volume'], color='black')
    plt.plot(averageinspbreath['poes'], averageinspbreath['volume'], color='black')
    plt.scatter(frc_x, frc)
    exp_elastic = plot_triangle_and_calculate_area((point_e),(point_c), (point_b), 'y', 'black')
    insp_elastic = plot_triangle_and_calculate_area((point_a),(point_c), (point_d), 'g', 'black')
    exp_res = exp_resistive(averageexpbreath, point_b, point_c, point_d)
    insp_res = insp_resistive(averageinspbreath, point_a, point_b, point_c, point_e)
    
    return insp_res, insp_elastic, exp_res, exp_elastic