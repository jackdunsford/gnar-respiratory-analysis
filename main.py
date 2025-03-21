import json
import gnar
import os

# Opening JSON file
with open('settings.json') as json_file:
    settings = json.load(json_file)

if __name__ == '__main__':
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)