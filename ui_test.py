import json
import os
import gnar
from nicegui import ui, app

# Path to the settings file
SETTINGS_FILE = "settings.json"

def load_settings():
    """Load settings from the JSON file."""
    try:
        with open(SETTINGS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    
def save_settings(settings):
    """Save settings to the JSON file."""
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)

def get_ic_files(input_folder):
    """Get a list of .txt files in the 'ic' folder."""
    ic_folder = os.path.join(input_folder, "ic")
    if not os.path.exists(ic_folder):
        return []
    files = [f for f in os.listdir(ic_folder) if f.endswith(".txt")]
    files.sort()
    return files

def get_breath_files(input_folder):
    breaths_folder = os.path.join(input_folder, 'breaths')
    if not os.path.exists(breaths_folder):
        return []
    files = [f for f in os.listdir(breaths_folder) if f.endswith(".txt")]
    files.sort()
    return files

def run(settings):
    print("Analyzing " + os.path.basename(settings['inputfolder']))
    output_df = gnar.analyse(settings)


def create_editor(settings):
    """Create a NiceGUI interface to edit the settings."""
    ui.label("Edit Settings").classes("text-h4")
    
    # Create a row to hold two columns
    with ui.row():
        # First column: General settings
        with ui.column():
            ui.label("General Settings").classes("text-h5")
            inputs = {}
            for key, value in settings.items():
                if key == "ignoreic":
                    continue
                if key =='ignorebreath':
                    continue
                if isinstance(value, bool):
                    inputs[key] = ui.checkbox(key, value=value)
                elif isinstance(value, int):
                    inputs[key] = ui.number(key, value=value)
                else:
                    inputs[key] = ui.input(key, value=str(value))
        
        # Second column: Ignore IC settings
        with ui.column():
            ui.label("Ignore IC Settings").classes("text-h5")
            
            # Check if inputfolder is set
            if "inputfolder" not in settings:
                ui.label("Please set 'inputfolder' in General Settings first.")
                return
            
            # Get .txt files from the 'ic' folder
            ic_files = get_ic_files(settings["inputfolder"])
            if not ic_files:
                ui.label("No .txt files found in the 'ic' folder.")
                return
            
            # Create a nested list for ignoreic
            ignoreic_inputs = {}
            for file_name in ic_files:
                # ui.label(f"File: {file_name}")
                ignoreic_inputs[file_name] = ui.input(
                    f"{file_name}",
                    placeholder="Enter numbers separated by commas (e.g., 2,4,6)",
                    value=",".join(map(str, next((item[1] for item in settings.get("ignoreic", []) if item[0] == file_name), [])))
                ).classes("w-full")
        # Third column: Ignore breath settings
        with ui.column():
            ui.label("Ignore breaths Settings").classes("text-h5")
            
            # Check if inputfolder is set
            if "inputfolder" not in settings:
                ui.label("Please set 'inputfolder' in General Settings first.")
                return
            
            # Get .txt files from the 'breath' folder
            breath_files = get_breath_files(settings["inputfolder"])
            if not breath_files:
                ui.label("No .txt files found in the 'breaths' folder.")
                return
            
            # Create a nested list for ignorebreath
            ignorebreaths_inputs = {}
            for file_name in breath_files:
                # ui.label(f"File: {file_name}")
                ignorebreaths_inputs[file_name] = ui.input(
                    f"{file_name}",
                    placeholder="Enter numbers separated by commas (e.g., 2,4,6)",
                    value=",".join(map(str, next((item[1] for item in settings.get("ignorebreath", []) if item[0] == file_name), [])))
                ).classes("w-full")
    # # Save button
    def on_save():
        updated_settings = {}
        for key, input_field in inputs.items():
            if isinstance(settings[key], bool):
                updated_settings[key] = input_field.value
            elif isinstance(settings[key], int):
                updated_settings[key] = int(input_field.value)
            else:
                updated_settings[key] = input_field.value
        
        # Update ignoreic setting
        updated_ignoreic = []
        for file_name, input_field in ignoreic_inputs.items():
            numbers = [int(num.strip()) for num in input_field.value.split(",") if num.strip()]
            if numbers:
                updated_ignoreic.append([file_name, numbers])
        updated_settings["ignoreic"] = updated_ignoreic
        
        # Update ignorebreath setting
        updated_ignorebreaths = []
        for file_name, input_field in ignorebreaths_inputs.items():
            numbers = [int(num.strip()) for num in input_field.value.split(",") if num.strip()]
            if numbers:
                updated_ignorebreaths.append([file_name, numbers])
        updated_settings["ignorebreath"] = updated_ignorebreaths

        save_settings(updated_settings)
        ui.notify("Settings saved successfully!")
    
    ui.button("Save", on_click=on_save).classes("mt-4")
    

settings = load_settings()
create_editor(settings)

ui.run()
