import gnar
import os

settings = {
    "inputfolder": r'/Users/jackdunsford/Desktop/02-119',
    
    "timecol":0,
    "flowcol":17,
    "volumecol":18,
    "poescol":13,   
    "pgascol":14,
    #enter the name of the file and the breath number you wish to exclude from the IC volume drift correction
    "ignoreic": [
        # ["01-101_V3_IC160.txt", [7,9,10,11]]
    ],
    #enter the name of the file and the breath number you wish to exclude from volume drift correction and averaging
    "ignorebreath":[
        ["01-101_V3_B080.txt", [2,3,4,5]],
        ["01-101_V3_B140.txt", [1,7,8]],
        ["01-101_V3_B100.txt", [4,5,6]],
        ["01-101_V3_B120.txt", [1,2,7,8,10]]
    ],
    "saveiccorrection": True, #saves a plot with the ic trend correction to check for incorrect EELV detection
    "saverawflowvolume": True, #saves a plot of the raw flow and volume traces
    "saveflowvolumeloops": True, #saves a plot with the FV loop and MEFV for each stage
    "savewobplots": True,
    "saveoutput": True, #saves an excel of the dataframe containing data for each stage of exercise
    "savemefv": True, #saves a figure of the MEFV curve with all FVCs
    "savemefvdata": True, #saves an excel with fvc, fev1, pef, slope ratio, and data for the MEFV curve
    "saveaveragedata": True, #saves an excel with the data of the average flow volume loop for each stage
    
    "age": 34,
    "sex": 1
}

if __name__ == '__main__':
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)