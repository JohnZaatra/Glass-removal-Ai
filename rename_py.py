import pandas as pd
import os
from PIL import Image
import pytesseract
from datetime import datetime
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# Configure the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Directory containing your GIF files
# directory = '/Users/john-zaatri/Desktop/test_on_image'
directory = '/Users/john-zaatri/Desktop/OP2-images/2018/5'

# Directory to save cropped images
# cropped_images_directory = '/Users/john-zaatri/Desktop/test_on_image/cropped_imgs'
cropped_images_directory = '/Users/john-zaatri/Desktop/test_on_image/crop'
# Ensure the cropped images directory exists
os.makedirs(cropped_images_directory, exist_ok=True)

# Define the file path
file_path = "/Users/john-zaatri/Desktop/semester 8/Final Project/PROJECT/data/processed_data_with_date.csv"

# Function to load CSV file
def load_csv(file_path):
    df = pd.read_csv(file_path)
    print("CSV file loaded successfully.")
    return df

# Function to extract unique sorted list from DataFrame
def extract_unique_sorted_list(df):
    unique_sorted_values = sorted(df['File'].unique())
    unique_sorted_list = list(unique_sorted_values)
    return unique_sorted_list

# Function to extract identification code
def extract_identification_code(text):
    if "Identification code:" in text:
        code = text.split("Identification code:")[-1].split()[0]
        return code
    return None

# Function to convert month name to number
def month_name_to_number(month_name):
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12"
    }
    return months.get(month_name.lower(), None)

# Function to extract and format date
def extract_date(text):
    # Check for Acquisition date
    if "Acquisition date:" in text:
        date_str = text.split("Acquisition date:")[-1].split()[0]
    elif "Exam date and time:" in text:
        date_str = text.split("Exam date and time:")[-1].split()[0]
    else:
        return None
    
    # Handle dates with month names
    parts = date_str.split("-")
    if len(parts) == 3 and month_name_to_number(parts[1]):
        parts[1] = month_name_to_number(parts[1])
        # Add prefix "20" to year if it has only two digits
        if len(parts[2]) == 2:
            parts[2] = "20" + parts[2]
        date_str = "-".join(parts)
    
    # Attempt to parse the date
    for fmt in ["%d/%m/%Y", "%d-%m-%Y"]:
        try:
            date = datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
            return date
        except ValueError:
            continue
    return None

# Function to crop and save images
def crop_and_save_images(img, code, date):
    # Define the crop areas for upper left and upper right (example values)
    upper_left_crop_area = (780, 100, 1180, 390)  # (left, top, right, bottom)
    upper_right_crop_area = (1413, 100, 1813, 390)  # (left, top, right, bottom)

    # Crop the images
    upper_left_image = img.crop(upper_left_crop_area)
    upper_right_image = img.crop(upper_right_crop_area)

    # Replace slashes in the date with underscores to avoid directory issues
    safe_date = date.replace("/", "-")

    # Define the new filenames
    upper_left_filename = f"{code}_{safe_date}_Tangential_anterior.gif"
    upper_right_filename = f"{code}_{safe_date}_Tangential_posterior.gif"

    # Save the cropped images
    upper_left_image.save(os.path.join(cropped_images_directory, upper_left_filename))
    upper_right_image.save(os.path.join(cropped_images_directory, upper_right_filename))

    print(f"Saved cropped images: {upper_left_filename} and {upper_right_filename}")

# Iterate over each file in the directory
# Function to process each image
def process_image(filename, df, unique_sorted_list):
    if filename.endswith(".gif"):
        image_path = os.path.join(directory, filename)

        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            code = extract_identification_code(text)
            date = extract_date(text)

            if code:
                try:
                    code_int = int(code)
                except ValueError:
                    print(f"Extracted code {code} is not a valid integer. Skipping image.")
                    return

                index = bisect_left(unique_sorted_list, code_int)

                if index < len(unique_sorted_list) and unique_sorted_list[index] == code_int:
                    relevant_row = df[df['File'] == code_int]
                    surgery_date = relevant_row['Surgery Date'].iloc[0]

                    if surgery_date and date:
                        if datetime.strptime(surgery_date, "%d/%m/%Y") >= datetime.strptime(date, "%d-%m-%Y"):
                            crop_and_save_images(img, code, date)
                        else:
                            print(f"Surgery Date {surgery_date} is before the date extracted from the image {date}. Skipping image with code {code}.")
                    else:
                        print(f"Missing Surgery Date or Date from image for code {code}. Skipping image.")
                else:
                    print(f"Code {code} not found in DataFrame. Skipping image.")
            else:
                print(f"Identification code not found in image {filename}. Skipping image.")
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Load CSV file once
    start_time = time.time()
    df = load_csv(file_path)

    # Extract unique sorted list once
    unique_sorted_list = extract_unique_sorted_list(df)

    # Process images using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(process_image, os.listdir(directory), [df]*len(os.listdir(directory)), [unique_sorted_list]*len(os.listdir(directory)))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time} seconds")



# Iterate over each file in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".gif"):  # Check if the file is a GIF
#         image_path = os.path.join(directory, filename)

#         # Open the image
#         img = Image.open(image_path)

#         # Use pytesseract to do OCR on the image
#         text = pytesseract.image_to_string(img)

#         # Extract identification code
#         code = extract_identification_code(image_path)
#         # Extract date
#         date = extract_date(text)

#         if code and date:
#             # Crop and save the images
#             crop_and_save_images(img, code, date)
#         else:
#             print(f"Missing identification code or date in file: {filename}")
