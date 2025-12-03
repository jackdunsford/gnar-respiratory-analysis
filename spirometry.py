import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def get_fvc(mefv):
    vital_capacity = mefv['volume'].iloc[-1]
    return vital_capacity

def get_peak_flow(mefv):
    peak_flow = mefv['flow'].max()
    return peak_flow

def get_fev1(mefv):
    mefv['time'] = (0.02 / mefv['flow']).cumsum().round(2)
    mefv['time'] = mefv['time'].values[::-1]
    fev1 = mefv.loc[mefv['time'] <= 1.00, 'volume'].iloc[0]
    
    return fev1

def get_slope_ratio(mefv):
    mefv_sr = mefv.copy()
    mefv_sr['percent'] = mefv_sr.volume / mefv_sr.volume.max()
    mefv_sr['volume'] = mefv_sr['volume'].values[::-1]
    count = 0
    sr_sum = 0
    for i in range(len(mefv_sr)-1):
        point = mefv_sr['percent'].iloc[i]
        if 0.20 <= point <= 0.80:
            flow_above = mefv_sr.loc[mefv_sr['volume'] == (round(mefv_sr['volume'][i]+0.2, 2)), 'flow'].values[0]
            flow_below = mefv_sr.loc[mefv_sr['volume'] == (round(mefv_sr['volume'][i]-0.2, 2)), 'flow'].values[0]
            tan = (flow_below - flow_above) / 0.4
            chord = mefv_sr['flow'][i] / mefv_sr['volume'][i]
            sr = abs(tan / chord)
            sr_sum += sr
            count += 1
    slope_ratio = sr_sum / count
    return(slope_ratio)

def save_spirometry(mefv, path, settings): 
    fvc=get_fvc(mefv)
    # fev1=get_fev1(mefv)
    fev1 = 0
    peak_flow=get_peak_flow(mefv)
    slope_ratio=get_slope_ratio(mefv)
    mefv2 = mefv[['time', 'volume', 'flow']]
    data = [{'fvc':fvc, 'fev1':fev1, 'fev1/fvc': fev1/fvc, 'peak_flow':peak_flow, 'slope_ratio':slope_ratio}]
    df = pd.DataFrame(data)
    savefile = os.path.join(settings['inputfolder'], "output", "data", "SpirometryData.xlsx")
    with pd.ExcelWriter(savefile) as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        mefv2.to_excel(writer, sheet_name='MEFV', index=False)

def get_max_fvc(fvc_folder, settings):
    dl = os.listdir(fvc_folder)
    fvc = 0
    for f in dl:
        if f.endswith(".txt"):
            path_in = os.path.join(fvc_folder,f)
            data = pd.read_csv(path_in,
                            delimiter='\t')
            data = data.iloc[:, [settings['flowcol'], settings['volumecol']]]
            data.columns = ['flow', 'volume']
            data['time'] = np.arange(len(data['flow'])) / 2000
            data['volume'] = (data['volume'] - data['volume'][0]).round(2)
            data = data[data['volume'] >= 0]
            if data['volume'].iloc[-1] > fvc:
                fvc = data['volume'].iloc[-1]
    return fvc

def individual_fvc(input_path:str, fvc, settings):
    """
    fed each file from the mefv_curve function and creates a data frame of
    that MEFV curve
    """
    # print(input_path)
    data = pd.read_csv(input_path,
                            delimiter='\t')
    data = data.iloc[:, [settings['flowcol'], settings['volumecol']]]
    data.columns = ['flow', 'volume']
    data['time'] = np.arange(len(data['flow'])) / settings['samplingfrequency']
    # volume= data[settings['volumecol']].to_numpy()
    # # poes = breaths['poes'].to_numpy()
    # flow = data[settings['flowcol']].to_numpy()
    # time = data[settings['timecol']].to_numpy()
    
    data['volume'] = (data['volume'] - data['volume'][0]).round(2)
    data = data[data['volume'] >= 0]
    individual_mefv = data.groupby('volume').mean().reset_index()
    diff = fvc - individual_mefv['volume'].iloc[-1]
    
    individual_mefv['volume'] = (individual_mefv['volume'] + diff).round(2)
    # individual_mefv = df.drop(['time'], axis=1).reset_index()
    individual_mefv.volume = individual_mefv.volume.values[::-1]
    individual_mefv = individual_mefv.reset_index()
    return individual_mefv


def mefv_curve(path, settings):
    """
    indexes the folder of fvc.txt files and creates final mefv curve by taking the 
    highest flow at each increment of lung volume across all FVC manoevers
    """
    master_df = pd.DataFrame()
    dl = os.listdir(path)
    fig, ax = plt.subplots()
    fvc = get_max_fvc(path, settings)
    for f in dl:
        if f.endswith(".txt"):
            path_in = os.path.join(path,f)
            df = individual_fvc(path_in, fvc, settings)
            ax.plot(df.volume, df.flow, alpha=0.2, color = "gray")
            master_df = pd.concat([master_df, df])
    
    mefv = master_df.groupby('volume').max().reset_index()

    if settings['savemefv']:
        ax.set_xlabel("Volume (L)")
        ax.set_ylabel("FLow (L/s)")
        ax.plot(mefv.volume, mefv.flow, color = "black")
        ax.axhline(y=0, color="gray")
        savefile = os.path.join(settings['inputfolder'], "output", "figures", "MEFV.pdf")
        fig.savefig(savefile) 
        plt.close()
    if settings['savemefvdata']:
        save_spirometry(mefv, path, settings)

    return mefv
