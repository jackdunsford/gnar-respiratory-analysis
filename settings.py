import gnar

settings = {
    "inputfolder": r'/Users/jackdunsford/Library/CloudStorage/OneDrive-Personal/Thesis/01-220/V2 gnarpy',
    #enter the name of the file and the breath number you wish to exclude from the IC volume drift correction
    "ignoreic": [
        ["01-220_V2_IC220.txt", [12]]
    ],
    #enter the name of the file and the breath number you wish to exclude from volume drift correction and averaging
    "ignorebreath":[
        ["01-220_V2_IC220.txt", [12]]
    ],
    "saveiccorrection": True, #saves a plot with the ic trend correction to check for incorrect EELV detection
    "saverawflowvolume": True, #saves a plot of the raw flow and volume traces
    "saveflowvolumeloops": True, #saves a plot with the FV loop and MEFV for each stage
    "saveoutput": True, #saves an excel of the dataframe containing data for each stage of exercise
    "savemefv": True, #saves a figure of the MEFV curve with all FVCs
    "savemefvdata": True, #saves an excel with fvc, fev1, pef, slope ratio, and data for the MEFV curve
    "saveaveragedata": True #saves an excel with the data of the average flow volume loop for each stage
}


output_df = gnar.analyse(settings)
 