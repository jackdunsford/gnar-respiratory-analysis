import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy as sp
from os.path import join as pjoin
from scipy import signal
import seaborn as sns
import scipy as sp
import os
from matplotlib.backends.backend_pdf  import PdfPages
import spirometry
import glob

def check_dir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def ignorebreaths(inputfile, foric, settings):
    curfile =  os.path.basename(inputfile)
    
    if foric:
        d = dict(settings['ignoreic'])
    else: 
        d = dict(settings['ignorebreath'])
    
    if curfile in d:
        ib = [x-1 for x in d[curfile]]
        return ib
    
    else: return []

def correcttrend(volume, file):
        """corrects volume trend"""

        vol = volume.squeeze()
        peaks = signal.find_peaks((vol*-1), prominence=0.05, distance=0.25 * 1000)[0]
        f = sp.interpolate.interp1d(peaks, vol[peaks], 'linear', fill_value="extrapolate")
        peaksresampled = f(np.linspace(0, vol.size-1, vol.size))
        corvol = volume - peaksresampled
        
        return corvol


def correcttrendic(volume, input_path, pdf, settings):
    """
    corrects drift and trend for the IC breaths, does not include final expiration
    before the IC to account for participant changes in breathing 
    """
    vol = volume.squeeze()

    peaks = signal.find_peaks((vol), distance=500, prominence=0.25)[0]
    valleys = signal.find_peaks((vol*-1), distance=500, prominence=0.25)[0]
    peaks = peaks[:len(peaks)-1]
    ib = ignorebreaths(input_path, True, settings)
    if len(ib) > 0:
        peaks = np.delete(peaks, ib)

    z = np.polyfit(peaks, vol[peaks], 1)
    p = np.poly1d(z)

    peaksresampled = p(np.linspace(0, vol.size-1, vol.size))
    corvol = volume - peaksresampled
    
    if settings['saveiccorrection']:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        fig.suptitle("Trend correction and selection of IC volume for " + os.path.basename(input_path), fontsize=15)
        axes[0].plot(volume)
        axes[0].plot(peaks, vol[peaks], "x", markersize=20)
        axes[0].set_title("Uncorrected volume with end of expiration marked")
        axes[0].set_ylabel("Volume (L)")
        axes[1].plot(volume, linewidth=1.5)
        axes[1].plot(peaksresampled, linewidth=1.5)
        axes[1].set_title("Uncorrected volume with line of best fit for end expiratory volume")
        axes[1].set_ylabel("Volume (L)")
        axes[2].plot(corvol)
        axes[2].plot(np.zeros(corvol.size), '--', linewidth=1.5)
        axes[2].set_title("Corrected volyme with line of best fit set to 0")
        axes[2].set_ylabel("Volume (L)")

        for x in range(3):
            count=1
            yl = list(axes[x].get_ylim())
            for point in valleys:
                axes[x].axvline(x=point, color="grey", ls="--", lw=1)
                if count != len(valleys):
                    text = " #" + str(count)
                    # axes[x].text(point, yl[1]-((yl[1]-yl[0])*0.05), text, fontsize=8)   
                    axes[x].text(point, yl[0], text, fontsize=8)   
                    count+=1

        pdf.savefig()
        plt.close()
    
    return corvol

def get_ic(ic_path, pdf, settings):

    breaths = pd.read_csv(ic_path,
                            delimiter='\t')

    volume = breaths.iloc[:,settings['volumecol']].to_numpy()
    trendcorvol = correcttrendic(volume, ic_path, pdf, settings)
    corvol = trendcorvol

    peaks = signal.find_peaks((corvol*-1), prominence=0.25, distance=50)[0]
    
    ic = abs(corvol[peaks][-1])

    return ic

def averagebreaths(breath_path, pdf, settings):
    """
    takes time, flow and trend and drift corrected volume of a series of breaths as np.arrays and
    separates into individual inspired and expired breaths based on inspired and expired volume peaks.
    *** volume trace must start on inspiration and end on expiration ***    

    Returns an ordered dict with flow, volume, and time each breath separated into inspiration
    and expiration for each breath
    """
   
    totalbreaths = OrderedDict()    
    breaths = pd.read_csv(breath_path,
                            delimiter='\t')
   
    

    volumeraw = breaths.iloc[:,settings['volumecol']].to_numpy()
    # poes = breaths['poes'].to_numpy()
    flow = breaths.iloc[:,settings['flowcol']].to_numpy()
    time = breaths.iloc[:,settings['timecol']].to_numpy()

    volumeraw = volumeraw - volumeraw[0]
    
    volume = correcttrend(volumeraw, breath_path)
    timecol = np.arange(0, len(flow)-1, dtype=int) / 1000
    breathcnt = 0
    
    peak, _  = signal.find_peaks(volume, prominence=0.25, distance=500, width=0.001)
    valley, _ = signal.find_peaks(volume*-1, prominence=0.1, height=-0.01, distance=500, width=0.001)
    ib = ignorebreaths(breath_path, False, settings)
    
    if settings['saverawflowvolume']:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        fig.suptitle("Inputted flow and volume for " + os.path.basename(breath_path), fontsize=15)
        ax[0].plot(flow)
        ax[0].set_title("Raw Flow")
        ax[0].set_ylabel("Flow (L/s)")
        ax[1].plot(volumeraw)
        ax[1].set_title("Volume (Uncorrected)")
        ax[1].set_ylabel("Volume(L)")
        ax[2].plot(volume)
        ax[2].set_title("Volume (Corrected)")
        ax[2].set_ylabel("Volume (L)")
        ax[2].set_xlabel("Time (ms)")
        for x in range(3):
            count=1
            yl = list(ax[x].get_ylim())
            for point in valley:
                ax[x].axvline(x=point, color="grey", ls="--", lw=1)
                if count != len(valley):
                    text = " #" + str(count)
                    ax[x].text(point, yl[1]-((yl[1]-yl[0])*0.05), text, fontsize=8)
                    count+=1
            if len(ib) > 0:
                for breathno in ib:
                    ax[x].axvspan(valley[breathno], valley[breathno + 1], facecolor='gray', alpha=0.2)


        pdf.savefig()
        plt.close()
    
    # for each peak in the volume trace, separates into inspiration and expiration
    for point in range(len(valley)-1):
        breathcnt += 1
        exp = {'time':(timecol[valley[point]:peak[point]]).squeeze(), 
               'flow':(flow[valley[point]:peak[point]]).squeeze(),
               'volume':(volume[valley[point]:peak[point]]).round(3).squeeze()}
        insp = {'time':(timecol[peak[point]:valley[point+1]+1]).squeeze(),
                'flow':(flow[peak[point]:valley[point+1]]).squeeze(),
                'volume':(volume[peak[point]:valley[point+1]]).round(3).squeeze()}
        # enters everything into a ordered dict by breath number

        individualbreath = OrderedDict([('number', breathcnt),
                             ('name','Breath #' + str(breathcnt)), 
                             ('expiration', exp),
                             ('inspiration', insp), 
                             ('time', np.concatenate((insp["time"], exp["time"])).squeeze()), 
                             ('flow', np.concatenate((insp["flow"], exp["flow"])).squeeze()),
                             ('volume', np.concatenate((insp["volume"], exp["volume"])).squeeze()),
                             ('breathcnt', breathcnt)])
        if breathcnt not in ib:
            totalbreaths[breathcnt] = individualbreath
        # else: print("excluding breath #" + str(breathcnt))        
    
    # print(len(totalbreaths))

    vt = []
    end_vol = []
    start_vol = []
    for breath in totalbreaths:
        vt.append(abs(totalbreaths[breath]['expiration']['volume'][-1]))
        end_vol = totalbreaths[breath]['expiration']['volume'][-1]
        start_vol = totalbreaths[breath]['expiration']['volume'][0]
    vt = sum(vt) / len(vt)
    
    end_vol = end_vol.mean()
    start_vol = start_vol.mean()
    
    inspiredbreath = pd.DataFrame()
    for breath in totalbreaths:
        vt1 = abs(totalbreaths[breath]['inspiration']['volume'][0])
        insp_df = pd.DataFrame({'flow': totalbreaths[breath]['inspiration']['flow'], 'volume': totalbreaths[breath]['inspiration']['volume']})
        insp_df['percent'] = (insp_df['volume'] / vt1).round(3)
        insp_df = insp_df.groupby(insp_df['percent']).mean().reset_index()
        insp_df.reset_index()
        inspiredbreath = pd.concat([inspiredbreath, insp_df])

    inspiredbreath = inspiredbreath.groupby(inspiredbreath['percent']).mean().reset_index()
    inspiredbreath['volume'] = inspiredbreath['percent'] * vt

    expiredbreath = pd.DataFrame()
    ex_df = pd.DataFrame()
    for breath in totalbreaths:
        vt2 = abs(totalbreaths[breath]['expiration']['volume'][-1])
        ex_df = pd.DataFrame({'flow': totalbreaths[breath]['expiration']['flow'], 'volume': totalbreaths[breath]['expiration']['volume'].round(2)})
        ex_df['percent'] = (ex_df['volume'] / vt2)
        ex_df['volume'].round(3)
        expiredbreath = pd.concat([expiredbreath, ex_df])
    
    expiredbreath.groupby(expiredbreath['percent']).mean().reset_index()
    expiredbreath['volume'] = (expiredbreath['percent'] * vt)

    te = []
    ti = []

    for breath in totalbreaths:
        breath_te = abs(totalbreaths[breath]['expiration']['time'][-1]) - abs(totalbreaths[breath]['expiration']['time'][0])
        te.append(breath_te)

    for breath in totalbreaths:
        breath_ti = abs(totalbreaths[breath]['inspiration']['time'][-1]) - abs(totalbreaths[breath]['inspiration']['time'][0])
        ti.append(breath_ti)
    
    

    exptime = (sum(te) / len(te)).round(2)
    insptime = (sum(ti) / len(ti)).round(2)

    fb = 60 / (exptime + insptime)
    
    ve = fb * vt

    expiredbreath['percent'] = expiredbreath['percent'].round(2)
    expiredbreath = expiredbreath.groupby(expiredbreath['percent']).mean().reset_index()
    inspiredbreath['percent'] = inspiredbreath['percent'].round(2)
    inspiredbreath = inspiredbreath.groupby(inspiredbreath['percent']).mean().reset_index()
    
    expiredbreath.loc[0, 'flow'] = 0
    expiredbreath.loc[expiredbreath.index[-1],'flow'] = 0
    inspiredbreath.loc[inspiredbreath.index[-1],'flow'] = 0
    inspiredbreath.loc[0, 'flow'] = 0
    expiredbreath = expiredbreath.reset_index()
    inspiredbreath = inspiredbreath.reset_index()
    return inspiredbreath, expiredbreath, exptime, insptime, fb, vt, ve

def get_efl_percent(mefv, avg_expired, avg_inspired, erv, filename, pdf, settings):
    
    count=0
    avg_expired.volume = avg_expired.volume.values[::-1]
    avg_expired.volume = avg_expired.volume + erv
    avg_inspired.volume = avg_inspired.volume + erv

    if settings['saveflowvolumeloops']:
        plt.plot(mefv['volume'], mefv['flow'])
        plt.plot(avg_expired['volume'], avg_expired['flow'])
        plt.plot(avg_inspired['volume'], avg_inspired['flow'])
        plt.axhline(y=0, color="gray")
        plt.xlabel("Volume (L)")
        plt.ylabel("Flow (L/s)")
        plt.title("MEFV and flow volume loops for " + os.path.basename(filename), fontsize=15)
        pdf.savefig()
        plt.close()

    if settings['saveaveragedata']:
        averagefv = pd.concat([avg_expired, avg_inspired])
        averagefv.to_excel(pjoin(settings['inputfolder'], "output", "data", "AverageFVloop.xlsx"), index=False)

    for i in range(len(avg_expired)):
        tvol = avg_expired.volume[i].round(2)
        tflow = avg_expired['flow'][i]
        if tvol in mefv.volume.values:
            mefv_flow = mefv[mefv['volume']==tvol].round(2)['flow'].values[0]
            if tflow > mefv_flow: count+=1
    
    efl_percent = ((count/len(avg_expired.flow))*100)
         
    if count >= 5: 
        efl = 1
    else: 
        efl = 0

    return efl, efl_percent

def get_vecap(mefv,vt,te,ti,erv,irv):
    mefv = mefv.reset_index()
    start = mefv.index[mefv['volume']==erv][0]
    end = mefv.index[mefv['volume']==irv][0]
    temax = 0
    for i in range(start, end):
        mef = mefv.flow[i]
        temax+= 0.01/mef

    ttotmax = temax/(te/(te+ti))
    fbmax = 60/ttotmax
    vecap = vt * fbmax
    

    return vecap

def mechanics(avginsp_df, avgexp_df, ic, mefv, vt, fb, ti, te, ve, filename, pdf, settings):
    
    avg_expired_efl = avgexp_df.copy()
    avg_inspired_efl = avginsp_df.copy()
    # avg_expired_vecap = avgexp_df.copy()

    fvc = spirometry.get_fvc(mefv).round(2)
    
    erv = (fvc - ic).round(2)
    irv = (erv + vt).round(2)

    efl, efl_percent = get_efl_percent(mefv, avg_expired_efl, avg_inspired_efl, erv, filename, pdf, settings)
    vecap = get_vecap(mefv,vt,te,ti,erv,irv)    
    
    
    
    mechanics = {'Fb': [fb.round(2)],
                 'VT': [vt.round(2)],
                 'VE': [ve.round(2)],
                 'IC': [ic.round(2)],
                 'ERV': [erv.round(2)],
                 'IRV': [irv.round(2)],
                 'Ti': [ti],
                 'Te': [te],
                 'VEcap': [vecap],
                 'VEcap(%)': [(ve / vecap)],
                 'EFL': [efl],
                 'EFL%': [efl_percent]}

    df = pd.DataFrame(mechanics)
    
    return df

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def analyse(settings):

    check_dir(pjoin(settings['inputfolder'], "output", "data"))
    check_dir(pjoin(settings['inputfolder'], "output", "figures"))

    inputfolder = settings['inputfolder']
    
    fvcfolder = pjoin(inputfolder, "fvc")
    outputfolder = pjoin(inputfolder, "output")
    
    breaths_dir = sorted(listdir_nohidden(pjoin(inputfolder, "breaths")))
    ic_dir = sorted(listdir_nohidden(pjoin(inputfolder, "ic")))

    mefv = spirometry.mefv_curve(fvcfolder, settings)

    outputdata = pd.DataFrame()
    for f in range(len(breaths_dir)):
        file_name = breaths_dir[f]
        if file_name.endswith(".txt"): 
            input_path = pjoin(inputfolder, "breaths", file_name)
        ic_file = ic_dir[f]
        if ic_file.endswith(".txt"):
            ic_input = pjoin(inputfolder, "ic", ic_file)  
        
        with PdfPages(pjoin(outputfolder, "figures", file_name +'_plots.pdf')) as pdf:
            ic = get_ic(ic_input, pdf, settings).round(2)
            avginsp_df, avgexp_df, te, ti, fb, vt, ve = averagebreaths(input_path, pdf, settings)
            df = mechanics(avginsp_df, avgexp_df, ic, mefv, vt, fb, ti, te, ve, file_name,  pdf, settings)
        
        outputdata = pd.concat([outputdata, df])
    
    if settings['saveoutput']:
        outputdata.to_excel(pjoin(outputfolder, "data", "ExerciseData.xlsx"), index=False)

    return outputdata
