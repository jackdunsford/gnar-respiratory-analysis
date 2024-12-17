# Grand Numerical Analysis of Respiration

## About GNAR
This version of Grand Numerical Analysis of Respiration or GNAR is a remake of the original "GNARx" program written in LabView, created to identify and quantify expiratory flow limitation during exercise. 
This program takes CPET time series data from data acquisition softwares such as LabChart and separates/averages breaths for each stage of exercise and outputs MEFV and tFV loops as well as the presence and magnitude of EFL.

**GNAR pipeline:**
* Composite MEFV curve is created from graded FVC maneouvers pre and post exercise
* Volume drift is corrected 
* Breaths are separated and averaged
* IC is calculated to determine placement of exercise FV loops in the MEFV curve
* The presence of EFL is determined by the presence and magnitude of overlap of the MEFV curve and FV loop

Aspects of this code are adapted from ![RespMech](https://github.com/emilwalsted/respmech) which is an analysis tool for analysing respiratory mechanics such as work of breathing and diaphragm EMG recorded with an esophageal balloon catheter. I am currently working on integrating GNAR into the RespMech pipeline.

Please note, I am a researcher first, programmer second. I learned python for the purpose of this project so the code may be inefficient at times. Please reach out if you find issues/areas that can be improved or want assistance with your use case!


## Set up
Clone this repository and open with your preferred Python IDE (I use VScode) running Python 3.12+.

The **gnar.py** file contains the analysis code.

The **spirometry.py** file is a module integrated into the gnar.py code but can function as a stand alone library for creation and analysis of MEFV curves.

The **settings.py** is where data folders will be inputted, analysis and output settings are adjusted, and where gnar.py is executed.


## Input
The program takes time series data of time, flow, and volume collected using data acquisition software (code is designed to work seamlessly with LabChart exported .txt files). Export data into the folders specified below with two .txt files for each stage (one in "breaths" and one in "ic") and as many FVC manoevers as collected.
### Input folder
The input folder must match the example folder provided in the repo, the folder name is typically the participant/study ID and contains three sub folders where the time series .txt files will be located and an 'output' folder with sub folders "data" and "figures".

To allow for the code to run each stage of exercise in order, use sequential 3 digit numbering when naming the the breaths and ic files (I use the watts for that stage of exercise). For example, participantID_B020W.txt or participantID_IC160W.txt.

**Time series data (export all channels):**

* The "breaths" folder should contain ~30 seconds of fairly clean breaths towards the end of that stage of exercise (ideally last 30 seconds). Do not include the IC breath in this. Start selection on an inspiration and end on an expiration as shown. ![here](https://github.com/jackdunsford/gnar-respiratory-analysis/blob/main/instructional_images/breath.png)

* The "ic" folder should contain ~30 seconds of fairly clean breaths AND the IC breath, starting on an inspiration and ending mid way on the expired breath directly following the IC breatg as shown. ![here](https://github.com/jackdunsford/gnar-respiratory-analysis/blob/main/instructional_images/ic.png)

* The "fvc" folder should contain a number of graded fvc manoevers before and after exercise (I used 8 pre and 8 post). Select from the lowest point on the volume trace (should be the zero crossing on flow trace) to the highest point on the volume trace (flow should again be 0) as shown. ![here](https://github.com/jackdunsford/gnar-respiratory-analysis/blob/main/instructional_images/ic.png)

## Running the code

Duplicate **settings.py** file (a new one for each participant or study or use duplicated as a master), add the path to your input folder as such:

'''
"inputfolder": r'/path/to/your/data/folder',
'''

Adjust settings as desired and run.

Data and figures will be saved to the "output" folder, review these to determine which breaths should be ignored upon second analysis and add these files and breaths to the "ignoreic" and "ignorebreath" as shown in the example.

Rerun the analysis and repeat until breath selection is clean.

## Output
By default, the code will output three figures for each stage, IC breath correction (ignoreic in settings will tell the code to ignore erroneous peaks), raw flow/volume and corrected volume for the breaths, and the FV loop inside the MEFV curve.

The code will also output four excel files with the processed/averaged data for each stage, average FV loop for each stage, the composite MEFV curve, and spirometry data based on the MEFV curve.

## Analytical details
### MEFV curves
Using graded FVC manoevers before and after exercise, the mefv_curve function in **spirometry.py** creates a composite MEFV curve by taking the highest flow achieved at each lung volume to account for bronchodilation and thoracic gas compression.
### IC drift correction
The code corrects for drift in the volume signal by adjusting a line of best fit through end expiratory points to 0. Erroneous breaths can be ignored from line of best fit calculation in **settings.py** by writing the file name and what breath number should be ignored.

## Future directions
* Work of breathing measurements using esophageal and transdiaphragmatic pressures using pressure-volume integration, Hedstrand, Otis, and Modified Campbell methods.
* Diaphragm EMG analysis including removal of ECG atrifacts
