import gnar
import os

settings = {
    "inputfolder": r'/Users/jackdunsford/Desktop/NITRATE 101',
    
    "timecol":0, 
    "flowcol":17,
    "volumecol":21,
    "poescol":9, #if no catheter data put "" instead of the column number  
    "pgascol":10,
    #enter the name of the file and the breath number you wish to exclude from the IC volume drift correction
    "ignoreic": [
        
    ],
    #enter the name of the file and the breath number you wish to exclude from volume drift correction and averaging
    "ignorebreath":[
        ["NITRATE101V1_B090.txt", [2,3]]
    ],

    "samplingfrequency": 2000,
    "peakdistance": 1000, #minimum distance between peaks, should be ~500ms, i.e. take into account sampling freqency 
    "peakprominence": 0.11,
    "workrateincrement": 20,

    "saveiccorrection": True, #saves a plot with the ic trend correction to check for incorrect EELV detection
    "saverawflowvolume": True, #saves a plot of the raw flow and volume traces
    "saveflowvolumeloops": True, #saves a plot with the FV loop and MEFV for each stage
    "savewobplots": True,
    "saveoutput": True, #saves an excel of the dataframe containing data for each stage of exercise
    "savemefv": True, #saves a figure of the MEFV curve with all FVCs
    "savemefvdata": True, #saves an excel with fvc, fev1, pef, slope ratio, and data for the MEFV curve
    "saveaveragedata": True, #saves an excel with the data of the average flow volume loop for each stage
    
    "age": 79,
    "sex": 1,
    "fvc": 3.92
}

if __name__ == '__main__':
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)