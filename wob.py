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

def get_points(avgexp_df, avginsp_df, frc_y, ccw_slope):

    #Define end insp and exp points
    y_eelv = avgexp_df['volume'].iloc[0]
    y_eilv = avgexp_df['volume'].iloc[-1]
    x_eelv = avgexp_df['poes'].iloc[0]
    x_eilv = avgexp_df['poes'].iloc[-1]

    #Calculate and plot the lung compliance line (black)
    cl_y = np.linspace(y_eilv, y_eelv, 100)
    cl_x = np.linspace(x_eilv, x_eelv, 100)
    cl_m = (y_eelv - y_eilv)/(x_eelv - x_eilv)

    #Extrapolate Cl to frc 
    # frv_y = frc
    frc_x = x_eilv + ((frc_y - y_eilv)/cl_m)
    
    x_ccw_eilv = frc_x + ((y_eilv - frc_y)/ ccw_slope)
    x_ccw_eelv = frc_x + ((y_eelv - frc_y)/ ccw_slope)

    return x_eelv, x_eilv, y_eelv, y_eilv, cl_y, cl_x, cl_m, frc_x, x_ccw_eilv, x_ccw_eelv

def get_ccw_slope(settings):
    age = settings['age']
    sex = settings['sex']
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
    return ccw_slope

def hedstrand(avgexp_df, avginsp_df, frc_y, ex_stage, settings):
    ccw_slope = get_ccw_slope(settings)
    x_eelv, x_eilv, y_eelv, y_eilv, cl_y, cl_x, cl_m, frc_x, x_ccw_eilv, x_ccw_eelv = get_points(avgexp_df, avginsp_df, frc_y, ccw_slope)
    insp_elastic = 0.5 * abs(x_eilv*(y_eelv - y_eelv) + x_eelv*(y_eelv - y_eilv) + x_eilv*(y_eilv - y_eelv))

    # Plot the triangle
    triangle_x = [x_eilv, x_eelv, x_eelv, x_eilv]
    triangle_y = [y_eilv, y_eelv, y_eilv, y_eilv]

    # plt.scatter([x1, x2, x3], [y1, y2, y3], color="red", zorder=5)
    hedstrand_fig = plt.figure(figsize=(5,7))
    plt.title("Hedstrand Diagram - " + str(ex_stage) + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(avgexp_df['poes'], avgexp_df['volume'], color='black')
    plt.plot(avginsp_df['poes'], avginsp_df['volume'],color='black')
    plt.fill(triangle_x[:-1], triangle_y[:-1], alpha=0.6, label="Filled Triangle", color='g', edgecolor='k')


    # insp_elastic = plot_triangle_and_calculate_area(point_a, point_b, (point_b[0], point_a[1]), 'green', 'black')

    exp_curve = np.array(list(zip(avgexp_df['poes'][1:], avgexp_df['volume'][1:])))
    # line_points = [point_b, (point_b[0], point_a[1])]
    # intersections = []

    intersection = find_intersection((x_eelv, y_eelv), (x_eelv, y_eilv), exp_curve)
    if isinstance(intersection, MultiPoint):
        intersections = [p for p in intersection.geoms][-1]
    else: intersections = intersection


    exp_y_curve = avgexp_df['volume'][avgexp_df['volume'] <= intersections.y]
    exp_x_curve = avgexp_df['poes'].iloc[:len(exp_y_curve)]
    exp_x_line = np.linspace(x_eelv, intersections.x, len(exp_x_curve))
    exp_res_curve = plt.fill_betweenx(exp_y_curve, exp_x_curve, exp_x_line,color='blue',alpha=0.6, edgecolor=None, zorder=1)
    exp_curve_area = 0
    for path in exp_res_curve.get_paths():
        vertices = path.vertices
        exp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))


    insp_y_curve = avginsp_df['volume']
    insp_x_curve = avginsp_df['poes']
    insp_x_line = np.linspace(x_eelv, x_eilv, len(insp_x_curve))
    insp_res_curve = plt.fill_betweenx(insp_y_curve, insp_x_curve, insp_x_line,color='red',alpha=0.6, edgecolor=None)
    insp_curve_area = 0
    for path in insp_res_curve.get_paths():
        vertices = path.vertices
        insp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    plt.close()
    return insp_curve_area, insp_elastic, exp_curve_area, hedstrand_fig

def plot_triangle_and_calculate_area(p1, p2, p3, col, edge):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the area using the determinant formula
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    #  the triangle
    triangle_x = [x1, x2, x3, x1]
    triangle_y = [y1, y2, y3, y1]

    # plt.scatter([x1, x2, x3], [y1, y2, y3], color="red", zorder=5)
    plt.fill(triangle_x[:-1], triangle_y[:-1], alpha=0.6, label="Filled Triangle", color=col, zorder=2)
    
    return area

def find_intersection(line_start, line_end, curve_points):
    # Create the line and curve as Shapely objects
    line = LineString([line_start, line_end])
    curve = LineString(curve_points)

    # Find the intersection between the line and the curve
    intersection = line.intersection(curve)

    return intersection 

def polypaths_area(polypaths):
    area = 0
    for path in polypaths.get_paths():
        vertices = path.vertices
        area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                 np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    return area 

def dh_campbell(avgexp_df, avginsp_df, frc_y, ex_stage, pdf, settings):
    """If ERV is above FRC (calculated as resting IC) use this version of the campbell diagram"""
    
    #get all points
    ccw_slope = get_ccw_slope(settings)
    x_eelv, x_eilv, y_eelv, y_eilv, cl_y, cl_x, cl_m, frc_x, x_ccw_eilv, x_ccw_eelv = get_points(avgexp_df, avginsp_df, frc_y, ccw_slope)

    #Plot the insp (blue) and exp (red) curves
    campbell_fig = plt.figure(figsize=(5,7))
    plt.title("Campbell Diagram - " + str(ex_stage) + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(avgexp_df['poes'], avgexp_df['volume'], color='r')
    plt.plot(avginsp_df['poes'], avginsp_df['volume'], color='b')
    # print(x_eelv, y_eelv)
    
    plt.plot(cl_x,cl_y, color='k')

    #Find and plot the chest wall compliance line
    ccw_y_line = np.linspace(frc_y, avginsp_df['volume'].max(), 100)
    ccw_x_line = (ccw_y_line - frc_y) / ccw_slope + frc_x
    plt.plot(ccw_x_line, ccw_y_line)

    #Find x point of Ccw at end insp and end exp
    x_ccw_eilv = frc_x + ((y_eilv - frc_y)/ ccw_slope)
    # x_ccw_eelv =

    #Find intersections between Ccw and exp curve
    exp_curve = np.array(list(zip(avgexp_df['poes'], avgexp_df['volume'])))
    intersection = find_intersection((frc_x, frc_y), (x_ccw_eilv, y_eilv), exp_curve)

    if not intersection.is_empty:
        intersecthigh = (intersection.geoms[0].x, intersection.geoms[0].y)
        intersectlow = (intersection.geoms[1].x, intersection.geoms[1].y)
        
        #Find expiratory resistive area only if there are intersects to 
        exp_y_curve = avgexp_df['volume'][(avgexp_df['volume'] >= intersectlow[1]) & (avgexp_df['volume']<= intersecthigh[1])]
        exp_x_curve = avgexp_df['poes'][(avgexp_df['volume'] >= intersectlow[1]) & (avgexp_df['volume']<= intersecthigh[1])]
        exp_x_line = np.linspace(intersectlow[0], intersecthigh[0], len(exp_y_curve))
        exp_res_curve = plt.fill_betweenx(exp_y_curve, exp_x_curve, exp_x_line,color='blue',alpha=0.6, edgecolor='none')
        exp_res_area = polypaths_area(exp_res_curve)
    else:  
        intersectlow = (x_ccw_eelv,y_eelv)
        exp_res_area = 0
    #Plot the the line parallel to Ccw representing PEEPi
    para_ccw_y_line = np.linspace(y_eelv, avginsp_df['volume'].max(), 100)
    para_ccw_x_line = (para_ccw_y_line - y_eelv) / ccw_slope + x_eelv
    # plt.plot(para_ccw_x_line, para_ccw_y_line)

    #Find the x value of parallel ccw line at eilv
    x_para_ccw_eilv = x_eelv + ((y_eilv - y_eelv)/ ccw_slope)

    
    #Find inspiratory resistive area
    insp_y_curve = avginsp_df['volume']
    insp_x_curve = avginsp_df['poes']
    insp_x_line = np.linspace(x_eelv, x_eilv, len(avginsp_df['poes']))
    insp_res_curve = plt.fill_betweenx(insp_y_curve, insp_x_curve, insp_x_line, color='red',alpha=0.6, edgecolor='none')

    #Find inspiratory elastic ara
    # insp_elastic_area = plot_triangle_and_calculate_area((x_eilv, y_eilv), (x_eelv, y_eelv), (x_para_ccw_eilv, y_eilv), 'g', None)

    insp_elastic_points = np.array([[x_eelv, y_eelv], [x_eilv, y_eilv], [x_ccw_eilv, y_eilv], [x_ccw_eelv, y_eelv]])
    x, y = insp_elastic_points[:, 0], insp_elastic_points[:, 1]
    insp_elastic_area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    plt.fill(insp_elastic_points[:, 0], insp_elastic_points[:, 1], 'green', alpha=0.5)
    
    
    #Find PEEPi area
    points = np.array([[x_eelv, y_eelv], [x_eelv, y_eilv], [x_ccw_eelv, y_eilv], [x_ccw_eelv, y_eelv]])
    # points = np.array([[x_eelv, y_eelv], [x_para_ccw_eilv, y_eilv], [x_ccw_eilv, y_eilv], [intersectlow[0], intersectlow[1]]])
    x, y = points[:, 0], points[:, 1]
    peepi_area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    plt.fill(points[:, 0], points[:, 1], 'yellow', alpha=0.5)
    


    
    insp_res_area = polypaths_area(insp_res_curve)
    plt.close()
    return insp_res_area, insp_elastic_area, peepi_area, exp_res_area, campbell_fig

def modified_cambell(avgexp_df, avginsp_df, frc, ex_stage, pdf, settings):
    ccw_slope = get_ccw_slope(settings)
    x_eelv, x_eilv, y_eelv, y_eilv, cl_y, cl_x, cl_m, frc_x, x_ccw_eilv, x_ccw_eelv = get_points(avgexp_df, avginsp_df, frc, ccw_slope)
    point_a = (x_eilv, y_eilv) #end insp
    point_b = (x_eelv, y_eelv) #end exp
    point_c = (frc_x, frc) #frc
    point_d = (x_ccw_eilv, y_eilv) #ccw end insp
    point_e = (x_ccw_eelv, y_eelv) #ccw end exp
    
    campbell_fig = plt.figure(figsize=(5,7))
    plt.title("Modified Campbell diagram - " + str(ex_stage) + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(avgexp_df['poes'], avgexp_df['volume'], color='black')
    plt.plot(avginsp_df['poes'], avginsp_df['volume'], color='black')
    # plt.scatter(frc_x, frc)
    
    #elastic components
    exp_elastic = plot_triangle_and_calculate_area((point_e),(point_c), (point_b), 'y', None)
    insp_elastic = plot_triangle_and_calculate_area((point_a),(point_c), (point_d), 'g', None)
    

    #expiratory resistive work
    # exp_res_triangle = plot_triangle_and_calculate_area()

    # exp_res = exp_resistive(avgexp_df, point_b, point_c, point_d)
    
    # insp_res = insp_resistive(avginsp_df, point_a, point_b, point_c, point_e)
    insp_curve = np.array(list(zip(avginsp_df['poes'], avginsp_df['volume'])))
    line_points = [point_a, point_c]

    insp_intersection = find_intersection(point_e, point_d, insp_curve)
    if isinstance(insp_intersection, MultiPoint):
        insp_intersections = [p for p in insp_intersection.geoms][-1]
    else: insp_intersections = insp_intersection
    
    if insp_intersections.is_empty:
        insp_res_tri = plot_triangle_and_calculate_area((point_a), (point_b), (point_c), 'red', 'none')
        y_curve = avginsp_df['volume']
        x_curve = avginsp_df['poes']
        x_line = np.linspace(point_b[0], point_a[0], len(x_curve))
        insp_res_curve = plt.fill_betweenx(y_curve, x_curve, x_line,color='red',alpha=0.6, edgecolor='none')
        insp_res_curve_area = polypaths_area(insp_res_curve)
    else:
        insp_res_tri = plot_triangle_and_calculate_area((insp_intersections.x, insp_intersections.y), (point_c), (point_a), 'red', 'none')
        y_curve = avginsp_df['volume'][avginsp_df['volume'] >= insp_intersections.y]
        x_curve = avginsp_df['poes'].iloc[len(avginsp_df['poes'])-len(y_curve):]
        x_line = np.linspace(insp_intersections.x,point_a[0], len(x_curve))
        insp_res_curve = plt.fill_betweenx(y_curve, x_curve, x_line,color='red',alpha=0.6, edgecolor='none', zorder=1)
        insp_res_curve_area = polypaths_area(insp_res_curve)
    insp_res_area = insp_res_curve_area + insp_res_tri
    
    
    exp_curve = np.array(list(zip(avgexp_df['poes'], avgexp_df['volume'])))

    intersection = find_intersection(point_c, point_d, exp_curve)
    if isinstance(intersection, MultiPoint):
        intersections = [p for p in intersection.geoms][0]
    else: intersections = intersection

    exp_res_tri = plot_triangle_and_calculate_area((intersections.x, intersections.y), (point_c), (point_b), 'blue', None)
    y_exp_curve = avgexp_df['volume'][avgexp_df['volume'] <= intersections.y]
    x_exp_curve = avgexp_df['poes'].iloc[:len(y_exp_curve)]
    x_exp_line = np.linspace(point_b[0], intersections.x, len(x_exp_curve))
    exp_res_curve = plt.fill_betweenx(y_exp_curve, x_exp_curve, x_exp_line, color='blue',alpha=0.6, edgecolor=None)
    exp_res_curve_area = polypaths_area(exp_res_curve)
    exp_res_area = exp_res_curve_area + exp_res_tri
    plt.close()
    
    return insp_res_area, insp_elastic, exp_res_area, exp_elastic, campbell_fig

def pvintegration(avgexp_df, avginsp_df, ex_stage):

    pv_fig = plt.figure(figsize=(5,7))
    plt.title("Pressure volume integration - " + str(ex_stage) + "W")
    plt.ylabel("Volume (L)")
    plt.xlabel("Esophageal Pressure (cmH2O)")
    plt.plot(avgexp_df['poes'], avgexp_df['volume'], color='black')
    plt.plot(avginsp_df['poes'], avginsp_df['volume'], color='black')
    # plt.show() 
    y_eelv = avgexp_df['volume'].iloc[0]
    y_eilv = avgexp_df['volume'].iloc[-1]
    x_eelv = avgexp_df['poes'].iloc[0]
    x_eilv = avgexp_df['poes'].iloc[-1]

    insp_y_curve = avginsp_df['volume']
    insp_x_curve = avginsp_df['poes']
    insp_x_line = np.linspace(x_eelv, x_eilv, len(insp_x_curve))
    insp_res_curve = plt.fill_betweenx(insp_y_curve, insp_x_curve, insp_x_line,color='red',alpha=0.6, edgecolor='none')
    insp_curve_area = 0
    for path in insp_res_curve.get_paths():
        vertices = path.vertices
        insp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    
    exp_y_curve = avgexp_df['volume']
    exp_x_curve = avgexp_df['poes']
    exp_x_line = np.linspace(x_eelv, x_eilv, len(exp_x_curve))
    exp_res_curve = plt.fill_betweenx(exp_y_curve, exp_x_curve, exp_x_line,color='blue',alpha=0.6, edgecolor='none')
    exp_curve_area = 0
    for path in exp_res_curve.get_paths():
        vertices = path.vertices
        exp_curve_area += 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) -
                                    np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    plt.close()
    return exp_curve_area, insp_curve_area, pv_fig

def work_of_breathing(avgexp_df, avginsp_df, erv, frc, fb, ex_stage, pdf, settings):
    output = pd.DataFrame()
    figs = []
    if settings['campbelldiagram']:
        if erv > frc:
            insp_res_area, insp_elastic_area, peepi_area, exp_res_area, campbell_fig = dh_campbell(avgexp_df, avginsp_df, frc, ex_stage, pdf, settings)
            exp_elastic_area = 0
        else:
            insp_res_area, insp_elastic_area, exp_res_area, exp_elastic_area, campbell_fig = modified_cambell(avgexp_df, avginsp_df, frc, ex_stage, pdf, settings)
            peepi_area = 0
        campbell_data = {
            'cb_IR': [round(insp_res_area* 0.09806, 3)],
            'cb_IE': [round(insp_elastic_area* 0.09806, 3)],
            'cb_ER': [round(exp_res_area* 0.09806, 3)],
            'cb_EE': [round(exp_elastic_area* 0.09806, 3)],
            'cb_PEEP': [round(peepi_area* 0.09806, 3)]
            }
        if output.empty:
            output = pd.DataFrame(campbell_data)
        else:
            output = pd.concat([output, pd.DataFrame(campbell_data)], axis=1)
        figs.append(campbell_fig)
    if settings['hedstranddiagram']:
        insp_curve_area, insp_elastic, exp_curve_area, hedstrand_fig = hedstrand(avgexp_df, avginsp_df, frc, ex_stage, settings)
        hedstrand_data = {
            'h_IR': [round(insp_curve_area* 0.09806, 3)],
            'h_IE': [round(insp_elastic* 0.09806, 3)],
            'h_ER': [round(exp_curve_area* 0.09806, 3)]
        }
        if output.empty:
            output = pd.DataFrame(hedstrand_data)
        else:
            output = pd.concat([output, pd.DataFrame(hedstrand_data)], axis=1)
        figs.append(hedstrand_fig)
    if settings['pvintegration']:
        insp_wob, exp_wob, pv_fig = pvintegration(avgexp_df, avginsp_df, ex_stage)
        pv_data = {
            'pv_insp': [round(insp_wob* 0.09806, 3)],
            'pv_exp': [round(exp_wob* 0.09806, 3)]
        }
        if output.empty:
            output = pd.DataFrame(pv_data)
        else:
            output = pd.concat([output, pd.DataFrame(pv_data)], axis=1)
        figs.append(pv_fig)
    return output, figs
    