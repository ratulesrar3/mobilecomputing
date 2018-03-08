'''
Router: Netgear_7a:52:d
Ratul's phone: cc:08:8d:52:ab:cc
May's phone: 40:98:ad:96:74:96
Yuxi's phone: 40:4e:36:8b:b0:b3
Jess's phone: ec:9b:f3:cc:32:35

Chromecast: Google_64:38:38

Jess's computer: a4:5e:60:eb:ec:bf
Ratul's computer: e4:ce:8f:42:64:32
May's computer: f4:0f:24:10:14:73

Home?: Google_16:e8:55
Apple_10:14:73
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../packets.csv', encoding='ISO-8859-1')
train.dropna(inplace=True)
train['Second'] = np.ceil(train['Time'])
train.columns = ['Time', 'Source', 'Destination', 'Length', 'RSSI', 'Second']

train['RSSI'] = train['RSSI'].str.strip(' dBm').astype('float64')

devices = {'router':'Netgear_7a:52:d',
            'chromecast':'Google_64:38:38',
            'ratulphone':'Apple_52:ab:cc',
            'mayphone':'Apple_96:74:96',
            'yuxiphone':'Htc_8b:b0:b3',
            'jessmac':'Apple_eb:ec:bf',
            'maymac':'Apple_10:14:73'
            }

def find_device(mac):
    return train[train['Source'].str.contains(mac)].head(5)

def device_packet_stats(device_label):
    device = devices[device_label]
    source, dest = get_device_traffic_counts(device)
    s, d = get_device_traffic_counts(device, grouped=False)

    num_s, num_d = source.mean(), dest.mean()
    size_s, size_d = s['Length'].mean(), d['Length'].mean()
    rss_s, rss_d = s['RSSI'].mean(), d['RSSI'].mean()

    return {'device':device_label,
            'packets_received':num_s,
            'size_received':size_s,
            'rss_received':rss_s,
            'packets_sent':num_d,
            'size_sent': size_d,
            'rss_sent':rss_d}

def get_device_traffic_counts(device, rolling=False, grouped=True):
    data = []

    for call in ['Source', 'Destination']:

        packets = train[train[call].str.contains(device)]

        if grouped:
            packets = packets.groupby('Second').count()['Length']

            if rolling:
                packets = packets.rolling(window=60, win_type='triang').mean().dropna()

        data += [packets]

    return tuple(data)

def plot_device_traffic(device):
    source, destination = get_device_traffic_counts(device, rolling=True)
    plt.plot(source)
    plt.plot(destination)
    plt.savefig(device + '.png')
    plt.close('all')
