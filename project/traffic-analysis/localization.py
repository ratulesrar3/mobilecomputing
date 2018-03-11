# localization scripts


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as scopt


def localization(coords, c, g, x0, y0):
	'''
	ldpl function
	'''
	x1, y1 = coords
	return c + g * np.log10(((x1-x0)**2 + (y1-y0)**2)**0.5)


def c_fit(tx):
	popt, pcov = scopt.curve_fit(localization,
	                             (tx.loc_x,tx.loc_y),
	                             tx.RSSI,
	                             maxfev=250000,
	                             bounds=((-100,-5,-20,-20),(30,5,40,40)))
	return popt


def fit(df, mac_addr, room_len, room_wid):
    '''
    checks for mac addr in df and returns estimated coords
    '''
    tx = df[df['Source'] == mac_addr]
    time = list(set(df['Second']))
    y_loc = [(0, y*room_len/max(time)) for y in time]
    x_loc = [(x*room_wid/max(time)+5, room_len) for x in time]
    
    x_coords = [x for x,y in y_loc]
    y_coords = [y for x,y in y_loc]

    loc_dict = {'loc_x':x_coords,
    			'loc_y':y_coords,
    			'Second':time}

    loc_df = pd.DataFrame(loc_dict, columns=['loc_x','loc_y','Second'])
    tx = tx.merge(loc_df, on='Second', how='left')

    popts = c_fit(tx)
    return popts