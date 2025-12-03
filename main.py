import json
import gnar
import os

# Opening JSON file
json_file = r'/Users/jackdunsford/Desktop/PCD GNAR/PCD01_V1/PCD01_V1.json'
print(json_file)
with open(json_file) as json_file:
    settings = json.load(json_file)

if __name__ == '__main__':
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)