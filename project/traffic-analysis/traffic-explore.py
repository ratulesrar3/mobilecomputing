import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

'''
Yuxi router: Netgear_7a:52:d
Jess router: 9c:34:26:da:a3:c6, 9d:34:26:da:a3:ca

Ratul phone: cc:08:8d:52:ab:cc
May phone: 40:98:ad:96:74:96
Yuxi phone: 40:4e:36:8b:b0:b3
Jess phone: ec:9b:f3:cc:32:35

Jess computer: a4:5e:60:eb:ec:bf
Ratul computer: e4:ce:8f:42:64:32
May computer: f4:0f:24:10:14:73
Yuxi computer: c4:b3:01:8a:a0:2c

Chromecast: Google_64:38:38
Home?: Google_16:e8:55
Apple_10:14:73

Camera SSID: CH170C5-59842CF8953C-707143
'''

'''
baseline = pd.read_csv('baseline.csv', encoding='ISO-8859-1')
camera = pd.read_csv('camera.csv', encoding='ISO-8859-1')

intersection = {'Apple_52:ab:cc',
                'Apple_52:ab:cc (cc:08:8d:52:ab:cc) (TA)',
                'Apple_eb:ec:3f (a4:5e:60:eb:ec:3f) (TA)',
                'Apple_eb:ec:bf',
                'Apple_eb:ec:bf (a4:5e:60:eb:ec:bf) (BSSID)',
                'Apple_eb:ec:bf (a4:5e:60:eb:ec:bf) (TA)',
                'Apple_eb:ec:bf (a5:5e:60:eb:ec:bf) (TA)',
                'Apple_eb:ec:df (a4:5e:60:eb:ec:df) (TA)',
                'Htc_8b:b0:b3',
                'Htc_8b:b0:b3 (40:4e:36:8b:b0:b3) (TA)',
                'Htc_8b:b0:b3 (41:4e:36:8b:b0:b3) (TA)',
                'SamsungE_cc:32:35',
                'Shenzhen_ca:20:ad'}k
'''

devices = {'yuxirouter':'Netgear_7a:52:d',
            'jessrouter':'ArrisGro_da:a3:ca',
            'chromecast':'Google_64:38:38',
            'ratulphone':'Apple_52:ab:cc',
            'mayphone':'Apple_96:74:96',
            'yuxiphone':'Htc_8b:b0:b3',
            'jessmac':'Apple_eb:ec:bf',
            'maymac':'Apple_10:14:73',
            'camera':'Shenzhen_ca:20:ad'
            }

def preprocess(df):
    df.dropna(inplace=True)
    df['Second'] = np.ceil(df['Time'])
    df.columns = ['Time', 'Source', 'Destination', 'Length', 'RSSI', 'Second']
    df['RSSI'] = df['RSSI'].str.strip(' dBm').astype('float64')

def find_device(data, mac):
    return list(set(data[data['Source'].str.contains(mac)]['Source']))

def device_packet_stats(data, device_label, known=False):
    device = device_label
    if known:
        device = devices[device_label]

    source, dest = get_device_traffic_counts(data, device)
    s, d = get_device_traffic_counts(data, device, grouped=False)

    num_s, num_d = source.mean(), dest.mean()
    size_s, size_d = s['Length'].mean(), d['Length'].mean()
    rss_s, rss_d = s['RSSI'].mean(), d['RSSI'].mean()

    return {'device':device_label,
            'packets_received':num_d,
            'size_received':size_d,
            'rss_received':rss_d,
            'packets_sent':num_s,
            'size_sent': size_s,
            'rss_sent':rss_s}

def get_device_traffic_counts(data, device, rolling=False, grouped=True):
    traffic = []

    for call in ['Source', 'Destination']:

        packets = data[data[call].str.contains(device)]

        if grouped:
            packets = packets.groupby('Second').count()['Length']

            if rolling:
                packets = packets.rolling(window=60, win_type='triang').mean().dropna()

        traffic += [packets]

    return tuple(traffic)

def plot_device_traffic(data, device):
    source, destination = get_device_traffic_counts(data, device, rolling=True)
    plt.plot(source, color='#2ab74f')
    plt.plot(destination, color='#e05077')
    plt.savefig('actvity-plots/' + device + '.png')
    plt.close('all')

def get_top_devices(data, head):
    return list(data.groupby(['Source'])['Time'].agg({"count": len}).sort_values("count", ascending=False).head(head).reset_index()['Source'])
