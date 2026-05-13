import json
import gnar
import os

# Opening JSON file
json_file = r'/Volumes/T7/10 - EMGdi Study/EMGdi study data/208/208_MS.json'
print(json_file)
with open(json_file) as json_file:
    settings = json.load(json_file)

if __name__ == '__main__':
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)

    """
Settings defaults/examples
{
    "inputfolder":"path/to/input/folder",
    "flowcol":17, #LabChart channel for each input data. May have to increase or decrease by 1 column if exporting a time column
    "volumecol":18,
    "poescol":0,  #leave as 0 if not collected (WOB measures will be skipped)
    "pgascol":0,
    "emgscmcol":0, 
    "emgscacol":0,
    "emgdicols":[],
    "fvc": 0,
    "ignoreic":[ #breaths that you want to ignore for each IC file. ["file_name", [list,of,breath,#s]
        ["PCD18V1_IC040.txt", [5]],
        ["PCD18V1_IC100.txt", [6,7,11,12]]
    ],
    "ignorebreath":[ #breaths that you want to ignore for each IC file.
        ["PCD20V1_B000.txt", [2,3,5]],
        ["PCD20V1_B040.txt", [1,2,3]],
        ["PCD20V1_B100.txt", [3]],
        ["PCD20V1_B200.txt", [13,14,15,16]]
    ]
    "ignoreemg":[
        ["101_BT_40fvc_35mmhg.txt", [2,3,5]]
    ],
    
    "samplingfrequency": 1000, #sampling frequency on LabCbart
    "peakdistance": 250, #when detecting peaks to correct for volume drift. Default should be 1/4 of sampling frequency. i.e. 0.25 seconds between point detection. 
    "peakprominence": 0.1, #minimum prominence of volume peaks to e detected as a breath
    "workrateincrement": 20, #work rate increments for the test.
    "saveiccorrection": "True", #adjust what data and plots are saved
    "saverawflowvolume": "True", 
    "saveflowvolumeloops": "True", 
    "savewobplots": "True",
    "saveoutput": "True", 
    "savemefv": "True", 
    "savemefvdata": "True", 
    "saveaveragedata": "True", 
    "campbelldiagram": "True",  #pick which WOB method to use. For first time running data through, set all to false to make sure the flow and volume data is correct.
    "hedstranddiagram":"True", 
    "pvintegration":"True", 

    "ecgsubmethod": "", # "adjusted" creates three ECG templates for small, medium, large ECGs for traces that have obvious respiratory sinus arrythmias  
    "save_emg_proc_figures": "True",
    "saveemgoutput": "True",
    "ecgmethod": "basic",
    "shareecgpeaksch": 4, #pick which channel ECGs are detected from; if 0, ECGs will be detected per channel 
    "ecgwindowsize": 0.3
    "ecgdistance": 500, # minimum time (ms) between detected ECGs
    "ecgheight": 0.2, # minimum absolute height (mv) of detected ECGs
    "ecgprominence": 0.1, # minimum prominence of ECGs (height relative to neighbors)

    "id": "20", # participant ID, will be added to the output excel file
    "age":46,
    "sex":1    
}
"""