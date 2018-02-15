import pandas as pd
import numpy as np
import json
import scipy.optimize as scopt

def load_json(file):
    '''
    Loads txt file and returns dataframe using the json dict
    '''
    data = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            d = str(line).replace("'", '"')
            data = json.loads(d)

    return pd.DataFrame(data)
  
  def preprocess(df_list):
    '''
    Computes rolling average for RSS for each trace
    '''
    clean_dfs = []
    for df in df_list:
        df['rss'] = pd.to_numeric(df['rss'])
        df['mean_rss'] = df['rss'].rolling(window=5, win_type='triang').mean()
        df = df.dropna()
        clean_dfs.append(df)
    macs = list(set(df['mac']))
    
    return clean_dfs, macs
  
  def localization(coords, c, g, x0, y0):
    x1, y1 = coords
    return c + g * np.log10(((x1-x0)**2 + (y1-y0)**2)**0.5)
  
  def c_fit(tx):
    popt, pcov = scopt.curve_fit(localization,
                                 (tx.loc_x,tx.loc_y),
                                 tx.rss,
                                 maxfev=250000,
                                 bounds=((-60,2,-20,-20),(30,6,20,20)))
    return popt
  
  def fit(df_list, trace_num=0):
    '''
    Iterates through traces to estimate missing parameters
    '''
    clean_dfs, macs = preprocess(df_list)
    df = clean_dfs[trace_num]
    popts = []
    for mac in macs:
        tx = df[df['mac'] == mac]
        popts.append([str(mac)] + list(c_fit(tx)))
        
    return pd.DataFrame(popts, columns=['mac','c', 'gamma', 'x', 'y'])
  
  def go(df_list):
    results = []
    for i in range(len(df_list)):
        results.append(fit(df_list, i))
        
    res_df = pd.concat(results)
    return res_df
