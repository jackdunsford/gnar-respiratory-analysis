import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import scipy as sp
from os.path import join as pjoin
from scipy import signal
import scipy as sp
import os
from matplotlib.backends.backend_pdf  import PdfPages
import spirometry
import wob

def check_pressure_input(settings):
    if settings['pgascol'] != 0 and settings['poescol'] == 0:
        raise ValueError("Esophgageal pressure is required to calculate Pdi")

def check_breath_separation(endinsp_pts, endexp_pts, volume, filename, settings):
    if not len(endinsp_pts) == (len(endexp_pts) + 1):
        plt.figure(figsize=(15,10), )
        plt.title("Incorrect peak detection for " + filename)
        plt.xlabel("Observations")
        plt.ylabel("Volume (L)")
        plt.plot(volume)
        plt.plot(endexp_pts, volume[endexp_pts], 'x', markersize=10)
        plt.plot(endinsp_pts, volume[endinsp_pts], 'x', markersize=10)
        plt.savefig(pjoin(settings['inputfolder'], "output", "error.pdf"))
        raise ValueError("Breaths not separated correctly, review error.pdf in output folder and check peak detection in settings")

def add_dataframe_to_pdf(pdf, dataframe, title):
    fig, ax = plt.subplots(figsize=(10, 6))  # Set figure size
    ax.axis("tight")
    ax.axis("off")  # Remove axes

    # Add title if needed
    ax.set_title(title, fontsize=14, pad=20)

    # Create table from DataFrame
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))

    # Save figure to PDF
    pdf.savefig(fig)
    plt.close(fig)

def check_make_dir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def check_dir(settings):
    input_dir = settings['inputfolder']
    if os.path.isdir(input_dir):
        filelist=['breaths','ic','fvc', 'output']
        for file in filelist:
            if not os.path.isdir(os.path.join(input_dir, file)):
                print(os.path.join(input_dir, file))
                raise Exception("Missing folders from input path")

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

def trim(flow, volume, poes, pgas):
    if flow[0] >= 0:
        # startix = np.argmax(flow<0) + 250
        startix = np.argwhere(flow <0)[0]-50
    else: startix = 0
    # posflow = np.argwhere(flow<=0)
    if flow[-1] <= 0:
        # endix = posflow[:,0][len(posflow[:,0])-1]-50
        endix = np.argwhere(flow>0)[-1]+50
    else:endix=len(flow)
    return flow[startix:endix],volume[startix:endix],poes[startix:endix], pgas[startix:endix]

def correcttrend(volume, settings):
        """corrects volume trend"""

        vol = volume.squeeze()
        peaks = signal.find_peaks((vol*-1), prominence=settings['peakprominence'], distance=settings['peakdistance'])[0]
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

    peaks = signal.find_peaks((vol), prominence=settings['peakprominence'], distance=settings['peakdistance'])[0]
    valleys = signal.find_peaks((vol*-1), prominence=settings['peakprominence'], distance=settings['peakdistance'])[0]
    peaks = peaks[:len(peaks)-1]
    ib = ignorebreaths(input_path, True, settings)
    if len(ib) > 0:
        peaks = np.delete(peaks, ib)

    z = np.polyfit(peaks, vol[peaks], 1)
    p = np.poly1d(z)

    peaksresampled = p(np.linspace(0, vol.size-1, vol.size))
    corvol = volume - peaksresampled
    
    if settings['saveiccorrection'] and pdf:
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
            if len(ib) > 0:
                for breathno in ib:
                    axes[x].axvspan(valleys[breathno], valleys[breathno + 1], facecolor='gray', alpha=0.2)
                    

        pdf.savefig()
        plt.close()
    
    return corvol

def get_ic(ic_path, pdf, settings):

    breaths = pd.read_csv(ic_path, delimiter='\t')
    volume = breaths.iloc[:,settings['volumecol']].to_numpy()
    trendcorvol = correcttrendic(volume, ic_path, pdf, settings)
    corvol = trendcorvol
    peaks = signal.find_peaks((corvol*-1), prominence=settings['peakprominence'], distance=settings['peakdistance'])[0]
    ic = abs(corvol[peaks][-1])
    return ic

def get_rest_ic(rest_ic_path, settings):
    rest_ic = []
    for file in sorted(os.listdir(rest_ic_path)):
        if file.endswith(".txt"):
            path_in = os.path.join(rest_ic_path, file)
            rest_ic.append(get_ic(path_in, False, settings))
    return pd.DataFrame(rest_ic).mean()

def pressure_swings(pressure, endinsp_pts, ib):
    poes_swings = []
    for point in range(len(endinsp_pts)-1):
        breathno = point + 1
        if breathno not in ib:
            segment = pressure[endinsp_pts[point]:endinsp_pts[point+1]]
            minimum  = segment.min()
            maximum = segment.max()
            poes_swings.append(maximum - minimum)
    return statistics.mean(poes_swings)


def average_breath(path, erv, pdf, settings):
    check_pressure_input(settings)
    
    #load data
    df = pd.read_csv(path, 
                 delimiter='\t')
    flow = df.iloc[:, settings['flowcol']].to_numpy()
    volumeraw = df.iloc[:, settings['volumecol']].to_numpy()
    
    check_pressure_input(settings)
    if settings['poescol'] != 0:
        poes = df.iloc[:, settings['poescol']].to_numpy()
    else: poes = np.zeros(len(volumeraw)) 
    if settings['pgascol'] != 0:
        pgas = df.iloc[:, settings['pgascol']].to_numpy()
    else: pgas = np.zeros(len(volumeraw))
    pdi = pgas - poes
    
    #trims breaths to start on inspiration and end on expiration
    try:
        flow, volumeraw, poes, pgas = trim(flow, volumeraw, poes, pgas)
    except:
        print('i')
        if settings['saverawflowvolume']:
            fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 10), constrained_layout=True)
            fig.suptitle("Input flow and volume for " + os.path.basename(path), fontsize=15)
            ax[0].plot(flow)
            ax[0].set_title("Raw Flow")
            ax[0].set_ylabel("Flow (L/s)")
            ax[1].plot(poes)
            ax[1].set_title("Oesophageal Pressure")
            ax[1].set_ylabel("Peso (cmH2O)")
            ax[2].plot(volumeraw)
            ax[2].set_title("Volume (Uncorrected)")
            ax[2].set_ylabel("Volume(L)")
            ax[3].plot(volume)
            ax[3].set_title("Volume (Corrected)")
            ax[3].set_ylabel("Volume (L)")
            ax[3].set_xlabel("Time (ms)")
            pdf.savefig()
            plt.close()
    
    #correct volume
    volumeraw = volumeraw - volumeraw[0]
    volume = correcttrend(volumeraw, settings)
    volume, 
    #find in and exp peaks
    endinsp_pts, _ = signal.find_peaks(volume*-1, prominence=settings['peakprominence'], distance=settings['peakdistance'])
    endexp_pts, _ = signal.find_peaks(volume, prominence=settings['peakprominence'], distance=settings['peakdistance'])
    
    #check that correct peaks and troughs are being selected, should be one more trough than peak
    check_breath_separation(endinsp_pts, endexp_pts, volume, os.path.basename(path),settings)

    fb = len(endinsp_pts)/(len(volume[endinsp_pts[0]:endinsp_pts[-1]])/settings['samplingfrequency'])*60
    ib = ignorebreaths(path, False, settings)
    
    if settings['saverawflowvolume']:
      fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 10), constrained_layout=True)
      fig.suptitle("Input flow and volume for " + os.path.basename(path), fontsize=15)
      ax[0].plot(flow)
      ax[0].set_title("Raw Flow")
      ax[0].set_ylabel("Flow (L/s)")
      ax[1].plot(poes)
      ax[1].set_title("Oesophageal Pressure")
      ax[1].set_ylabel("Peso (cmH2O)")
      ax[2].plot(volumeraw)
      ax[2].set_title("Volume (Uncorrected)")
      ax[2].set_ylabel("Volume(L)")
      ax[3].plot(volume)
      ax[3].set_title("Volume (Corrected)")
      ax[3].set_ylabel("Volume (L)")
      ax[3].set_xlabel("Time (ms)")
      for x in range(4):
          count=1
          yl = list(ax[x].get_ylim())
          for point in endinsp_pts:
              ax[x].axvline(x=point, color="grey", ls="--", lw=1)
              if count != len(endinsp_pts):
                  text = " #" + str(count)
                  ax[x].text(point, yl[1]-((yl[1]-yl[0])*0.05), text, fontsize=8)
                  count+=1
          for point2 in endexp_pts:
            ax[x].axvline(x=point2, color="grey", ls="--", lw=0.5)
          if len(ib) > 0:
              for breathno in ib:
                  ax[x].axvspan(endinsp_pts[breathno], endinsp_pts[breathno + 1], facecolor='gray', alpha=0.2)
    pdf.savefig()
    plt.close()
    
    #calculates poes swings if a poes column is given
    if settings['poescol'] != 0:
        avg_poes_swing = pressure_swings(poes, endinsp_pts, ib)
        avg_pdi_swing = pressure_swings(pdi, endinsp_pts, ib)
    else:
        avg_poes_swing = np.float64(0.0)
        avg_pdi_swing = np.float64(0.0)
    avg_swings = [avg_poes_swing, avg_pdi_swing]        

    #separates expired from inspired breaths and takes the average based on % of tidal volume for each breath
    totalexpbreaths = pd.DataFrame(columns=['flow', 'volume', 'poes', 'percent'])
    expcount = 0
    for point in range(0, len(endinsp_pts)-1):
        if point not in ib:
            expcount+=1
            expbreath = pd.DataFrame({'flow':flow[endinsp_pts[point]:endexp_pts[point]].squeeze(),
                                'volume':volume[endinsp_pts[point]:endexp_pts[point]].squeeze(),
                                'poes':poes[endinsp_pts[point]:endexp_pts[point]].squeeze()[::-1],
                                'percent':volume[endinsp_pts[point]:endexp_pts[point]].squeeze()/(volume[endinsp_pts[point]:endexp_pts[point]]).squeeze()[-1]})
            mask = expbreath['flow'] < 0 
            expbreath = expbreath[~mask]
            totalexpbreaths = (expbreath.copy() if totalexpbreaths.empty else totalexpbreaths.copy() if expbreath.empty
        else pd.concat([expbreath, totalexpbreaths]) # if both DataFrames non empty
        )
        totalexpbreaths['percent'] = totalexpbreaths['percent'].round(2)
        averageexpbreath = totalexpbreaths.groupby(['percent']).mean().reset_index()


    totalinspbreaths = pd.DataFrame(columns=['flow', 'volume', 'poes', 'percent'])
    inspcount = 0
    for point in range(0, len(endinsp_pts)-1):
        if point not in ib:
            inspcount+=1
            inspbreath = pd.DataFrame({'flow':flow[endexp_pts[point]:endinsp_pts[point+1]].squeeze(),
                                'volume':volume[endexp_pts[point]:endinsp_pts[point+1]].squeeze(),
                                'poes':poes[endexp_pts[point]:endinsp_pts[point+1]].squeeze()[::-1],
                                'percent':volume[endexp_pts[point]:endinsp_pts[point+1]].squeeze()/(volume[endexp_pts[point]:endinsp_pts[point+1]]).squeeze()[0]})
            mask = inspbreath['flow'] > 0 
            inspbreath = inspbreath[~mask]
            totalinspbreaths = (inspbreath.copy() if totalinspbreaths.empty else totalinspbreaths.copy() if inspbreath.empty
        else pd.concat([inspbreath, totalinspbreaths]) # if both DataFrames non empty
        )
        totalinspbreaths['percent'] = totalinspbreaths['percent'].round(2)
        averageinspbreath = totalinspbreaths.groupby(['percent']).mean().reset_index()

    
    averageexpbreath['time'] = (averageexpbreath['volume'].diff() / averageexpbreath['flow']).cumsum()
    averageinspbreath['time'] = (averageinspbreath['volume'].diff() / abs(averageinspbreath['flow'])).cumsum()
    averageexpbreath.loc[0, 'time'] = 0
    averageinspbreath.loc[0, 'time'] = 0

    # averageexpbreath.loc[0, 'flow'] = 0
    # averageexpbreath.loc[averageexpbreath.index[-1], 'flow'] = 0
    # averageinspbreath.loc[0, 'flow'] = 0
    # averageinspbreath.loc[averageinspbreath.index[-1], 'flow'] = 0


    # averageexpbreath.loc[averageexpbreath.index[-1], 'volume'] = averageinspbreath['volume'].iloc[-1]
    # averageinspbreath.loc[0, 'volume'] = averageexpbreath['volume'].iloc[0]

    averageexpbreath = averageexpbreath.dropna()
    averageinspbreath = averageinspbreath.dropna()
    # if averageinspbreath.shape[0] != 
    #TODO fix alignment of volume/poes
    
    vt = averageexpbreath['volume'].iloc[-1]
    ve = fb * vt  

    te = averageexpbreath['time'].iloc[-1].round(3)
    ti = averageinspbreath['time'].iloc[-1].round(3)

    averageexpbreath['volume'] = averageexpbreath['volume'] + erv 
    averageinspbreath['volume'] = averageinspbreath['volume'] + erv 

    averageexpbreath['time'] = (averageexpbreath['volume'].diff() / averageexpbreath['flow']).cumsum()
    averageinspbreath['time'] = (averageinspbreath['volume'].diff() / abs(averageinspbreath['flow'])).cumsum()
    averageexpbreath.loc[0, 'time'] = 0
    averageinspbreath.loc[0, 'time'] = 0

    averageexpbreath.loc[0, 'flow'] = 0
    averageexpbreath.loc[averageexpbreath.index[-1], 'flow'] = 0
    averageinspbreath.loc[0, 'flow'] = 0
    averageinspbreath.loc[averageinspbreath.index[-1], 'flow'] = 0

    vol_start = (averageexpbreath['volume'].iloc[0] + averageinspbreath['volume'].iloc[0]) / 2
    vol_end = (averageexpbreath['volume'].iloc[-1] + averageinspbreath['volume'].iloc[-1]) / 2
    averageexpbreath.loc[0, 'volume'] = vol_start
    averageinspbreath.loc[0, 'volume'] = vol_start

    averageexpbreath.loc[averageexpbreath.index[-1], 'volume'] = vol_end
    averageinspbreath.loc[averageinspbreath.index[-1], 'volume'] = vol_end

    if settings['poescol'] != 0:
        poes_start = (averageexpbreath['poes'].iloc[0] + averageinspbreath['poes'].iloc[0]) / 2
        poes_end = (averageexpbreath['poes'].iloc[-1] + averageinspbreath['poes'].iloc[-1]) / 2
        averageexpbreath.loc[0, 'poes'] = poes_start
        averageinspbreath.loc[0, 'poes'] = poes_start

        averageexpbreath.loc[averageexpbreath.index[-1], 'poes'] = poes_end
        averageinspbreath.loc[averageinspbreath.index[-1], 'poes'] = poes_end

    return averageexpbreath, averageinspbreath, avg_swings, te, ti, fb, vt, ve

def get_efl_percent(mefv, avg_expired, avg_inspired, filename, settings):

    count=0
    avg_expired.volume = avg_expired.volume.values[::-1]
    avg_inspired.volume = avg_inspired.volume.values[::-1]

    # if settings['saveflowvolumeloops']:

    efl_fig = plt.figure(figsize=(10,10))
    plt.plot(mefv['volume'], mefv['flow'])
    plt.plot(avg_expired['volume'], avg_expired['flow'])
    plt.plot(avg_inspired['volume'], avg_inspired['flow'])
    _,xmax = plt.xlim()
    plt.axhline(y=0, xmin=0, xmax=xmax , color="gray")
    plt.xlabel("Volume (L)")
    plt.ylabel("Flow (L/s)")
    plt.title("MEFV and flow volume loops for " + os.path.basename(filename), fontsize=15)
    
        # pdf.savefig()
        # plt.close()

    if settings['saveaveragedata']:
        averagefv = pd.concat([avg_expired, avg_inspired])
        averagefv.to_excel(pjoin(settings['inputfolder'], "output", "data", "AverageFVloop.xlsx"), index=False)

    for i in range(len(avg_expired)):
        tvol = avg_expired.volume.iloc[i].round(2)
        tflow = avg_expired['flow'].iloc[i]
        if tvol in mefv.volume.values:
            mefv_flow = mefv[mefv['volume']==tvol].round(2)['flow'].values[0]
            if tflow > mefv_flow: count+=1
    
    efl_percent = ((count/len(avg_expired.flow))*100)
         
    if count >= 5: 
        efl = 1
    else: 
        efl = 0

    return efl, efl_percent, efl_fig

def get_vecap(mefv,vt,te,ti,erv,irv):
    mefv = mefv.reset_index()
    start = mefv.index[mefv['volume']==erv][0]
    end = mefv.index[mefv['volume']==irv][0]
    temax = 0
    for i in range(start, end):
        mef = mefv.flow[i]
        temax+= 0.01/mef
    ttotmax = temax/(ti/(te+ti))
    fbmax = 60/ttotmax
    vecap = vt * fbmax
    
    return vecap

def workofbreathing(avginsp_df, avgexp_df, fb, ex_stage, settings):
    x_eelv, x_eilv, y_eelv, y_eilv = wob.get_points(avgexp_df)
    point_a = (x_eilv, y_eilv) #end insp
    point_b = (x_eelv, y_eelv) #end exp

    insp_res, insp_elastic, exp_res, wob_fig  = wob.hedstrand(avgexp_df, avginsp_df, point_a, point_b, ex_stage, settings)

    insp_res_wob = insp_res * 0.09806 * fb
    insp_elas_wob = insp_elastic * 0.09806 * fb
    exp_res_wob = exp_res * 0.09806 * fb

    return insp_res_wob, insp_elas_wob, exp_res_wob, wob_fig


def mechanics(avginsp_df, avgexp_df, avg_swings, ic, mefv, te, ti, vt, fb, ve, frc, filename, ex_stage, pdf, settings):
    
    avg_expired_efl = avgexp_df.copy()
    avg_inspired_efl = avginsp_df.copy()

    fvc = spirometry.get_fvc(mefv).round(2)
    erv = (fvc - ic).round(2)
    irv = (erv + vt).round(2)

    print("\t\t\t Determining presence of EFL and saving FV loop")
    efl, efl_percent, efl_fig = get_efl_percent(mefv, avg_expired_efl, avg_inspired_efl, filename, settings)
    vecap = get_vecap(mefv,vt,te,ti,erv,irv)    
    

    # if settings['poescol'] != 0:
    #     print("\t\t\t Calculating work of breathing and saving the Hedstrand plot")

    #     if settings['savecampbelldiagram']:

    #     insp_res, insp_elastic, exp_res, wob_fig = workofbreathing(avginsp_df, avgexp_df, fb, ex_stage, settings)
    #     total_wob = insp_res + insp_elastic + exp_res
    # else: insp_res, insp_elastic, exp_res, total_wob = 0,0,0,0

    mechanics = {'Workrate': [ex_stage],
                 'FVC': [fvc.round(3)],
                 'Fb': [round(fb, 3)],
                 'VT': [vt.round(3)],
                 'VE': [ve.round(3)],
                 'IC': [ic.round(3)],
                 'ERV': [erv.round(3)],
                 'IRV': [irv.round(3)],
                 'poes_swing': [round(avg_swings[0], 3)],
                 'pdi_swing': [round(avg_swings[1],3)],
                 'VEcap': [vecap.round(3)],
                 'VEcap(%)': [((ve / vecap)*100).round(3)],
                 'EFL': [efl],
                 'EFL%': [efl_percent]}

    df = pd.DataFrame(mechanics)
    
    wob_output, wob_figs = wob.work_of_breathing(avgexp_df, avginsp_df, erv, frc, fb, ex_stage, pdf, settings)
    df = pd.concat([df, wob_output], axis=1)

    if len(wob_figs) >= 2:
        mosaic = """
        AABB
        CCDD
        """
        combined_fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 8), gridspec_kw={"height_ratios": [1,1]}, constrained_layout=True)
    else: 
        mosaic = """
        AABB
        """
        combined_fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 8), gridspec_kw={"height_ratios": [1]}, constrained_layout=True)
    # wob_fig = wob_figs[0]
    
    # Copy efl figure to ax1
    for line in efl_fig.axes[0].get_lines():
        axs["A"].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())
    for collection in efl_fig.axes[0].collections:  # Handles filled areas
        for path, color in zip(collection.get_paths(), collection.get_facecolor()):
            patch = plt.Polygon(path.vertices, closed=True, color=color, alpha=collection.get_alpha())
            axs["A"].add_patch(patch)
    for patch in efl_fig.axes[0].patches:  # Handles fill
        new_patch = plt.Polygon(patch.get_path().vertices, closed=True, facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(), alpha=patch.get_alpha())
        axs["A"].add_patch(new_patch)
    axs["A"].set_title(efl_fig.axes[0].get_title())
    axs["A"].set_ylabel(efl_fig.axes[0].get_ylabel())
    axs["A"].set_xlabel(efl_fig.axes[0].get_xlabel())

    spots = ["B", "C", "D"]
    for fig in range(len(wob_figs)):
        if settings['poescol'] != 0:
            # Copy hedstrand diagram to ax2 (if present)
            for line in wob_figs[fig].axes[0].get_lines():
                axs[spots[fig]].plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())
            for collection in wob_figs[fig].axes[0].collections:  # Handles filled areas
                for path, color in zip(collection.get_paths(), collection.get_facecolor()):
                    patch = plt.Polygon(path.vertices, closed=True, color=color, alpha=collection.get_alpha())
                    axs[spots[fig]].add_patch(patch)
            for patch in wob_figs[fig].axes[0].patches:  # Handles fill
                new_patch = plt.Polygon(patch.get_path().vertices, closed=True, facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(), alpha=patch.get_alpha())
                axs[spots[fig]].add_patch(new_patch)
            axs[spots[fig]].set_title(wob_figs[fig].axes[0].get_title())
            axs[spots[fig]].set_ylabel(wob_figs[fig].axes[0].get_ylabel())
            axs[spots[fig]].set_xlabel(wob_figs[fig].axes[0].get_xlabel())

    #add df to the ax3
    # axs["E"].axis("off")
    # table = axs["E"].table(
    #     cellText=df.values,
    #     colLabels=df.columns,
    #     loc="center",
    #     cellLoc="center",
    #     edges='open'
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(9)
    # table.auto_set_column_width(col=list(range(len(df.columns))))
    
    #save three panel figure to a pdf
    pdf.savefig(combined_fig)
    plt.close()
    return df

def listdir_nohidden(path):
    filenames = os.listdir(path)
    for f in filenames:
        if not f.startswith('.'):
            yield f


def analyse(settings):
    check_dir(settings)
    check_make_dir(pjoin(settings['inputfolder'], "output", "data"))
    check_make_dir(pjoin(settings['inputfolder'], "output", "figures"))

    inputfolder = settings['inputfolder']
    
    fvcfolder = pjoin(inputfolder, "fvc")
    outputfolder = pjoin(inputfolder, "output")
    breaths_dir = sorted(listdir_nohidden(pjoin(inputfolder, "breaths")))
    ic_dir = sorted(listdir_nohidden(pjoin(inputfolder, "ic")))

    increment = settings['workrateincrement']
    stage_count = 0
    print("\t Calculating MEFV")
    mefv = spirometry.mefv_curve(fvcfolder, settings)
    fvc = spirometry.get_fvc(mefv).round(2)

    outputdata = pd.DataFrame()
    for f in range(len(breaths_dir)):
        ex_stage = stage_count * increment
        file_name = breaths_dir[f]
        if file_name.endswith(".txt"):
            print("\t Loading " + file_name)
            ex_stage = str(ex_stage) 
            input_path = pjoin(inputfolder, "breaths", file_name)
        ic_file = ic_dir[f]
        if ic_file.endswith(".txt"):
            ic_input = pjoin(inputfolder, "ic", ic_file)  

        with PdfPages(pjoin(outputfolder, "figures", file_name +'_plots.pdf')) as pdf:
            print("\t\t Calculating IC")
            ic = get_ic(ic_input, pdf, settings).round(2)
            if stage_count == 0:
                frc = fvc - ic
            erv = fvc - ic
            print("\t\t Calculating average breath") 
            avgexp_df, avginsp_df, avg_swings, te, ti, fb, vt, ve = average_breath(input_path, erv, pdf, settings)
            
            print("\t\t Calculating breathing mechanics")
            df = mechanics(avginsp_df, avgexp_df, avg_swings, ic, mefv, te, ti, vt, fb, ve, frc, file_name, ex_stage, pdf, settings)
            # add_dataframe_to_pdf(pdf, df,file_name)
        stage_count+=1
        
        outputdata = pd.concat([outputdata, df])

        plt.close()
    if settings['saveoutput']:
        print("\t Saving all data")
        outputdata.to_excel(pjoin(outputfolder, "data",  "exercise_data.xlsx"), index=False)
        print("Analysis complete, find data and figures at: " + outputfolder)
    return outputdata
