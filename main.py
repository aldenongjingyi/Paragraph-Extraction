import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Initialise images from local folder into variable
raw_files = glob('00*.png')
print("Processing the following set of files: ")
print(raw_files)

# Filter out only project images
pages = []
file_names = ['001.png', '002.png', '003.png', '004.png', '005.png', '006.png', '007.png', '008.png']
for i in range(8):
    if raw_files[i] in file_names:
        pages.append(raw_files[i])

# Create the "Output" folder if it doesn't exist
output_folder = 'Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image file in the list
for image_path in raw_files:
    # Load the image
    page = cv2.imread(image_path)

    height, width, channels = page.shape

    # Convert image to grayscale
    gray_page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur_page = cv2.GaussianBlur(gray_page, (7, 7), 0)

    # Threshold the image to binarize  (inverse thresholding used so contours can be detected later)
    binary_page = cv2.threshold(blur_page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Access second returned value from cv2.threshold function

    # Dilate the paragraphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated_page = cv2.dilate(binary_page, kernel, iterations=5)  # 5 Iterations

    # Find contours and store in list (contours = potential paragraphs)
    contour_list = []
    contour_coordinates = []
    contour_list_binary = []
    contours, _ = cv2.findContours(dilated_page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found in page: " + str(len(contours)))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Extract ROI for normal
        contour = page[y: y + h, x: x + w]
        contour_list.append(contour)
        contour_coordinates.append((x, y))  # Storing contour coordinates for paragraph sorting
        # Extract ROI for binary
        contour_binary = binary_page[y: y + h, x: x + w]
        contour_list_binary.append(contour_binary)

    # Check for tables and images in contours
    paragraph_and_coordinates_u = list(zip(contour_list, contour_coordinates))
    paragraph_and_coordinates = []
    for c in range(len(contour_list_binary)):
        table_threshold = 40  # Determines number of consecutive pixels to assume contour is not text
        consecutive_white_pixels = 0
        is_text = True  # Flag to track if contour contains a table

        for row in range(contour_list_binary[c].shape[0]):
            for col in range(contour_list_binary[c].shape[1]):
                pixel_value = contour_list_binary[c][row, col]
                if pixel_value == 255:  # Check if the pixel is white
                    consecutive_white_pixels += 1
                    if consecutive_white_pixels >= table_threshold:
                        is_text = False
                        break
                else:
                    consecutive_white_pixels = 0

            if is_text == False:
                break

        if is_text:
            paragraph_and_coordinates.append(paragraph_and_coordinates_u[c])

    # Sort paragraphs in correct order
    sorted_paragraphs = sorted(paragraph_and_coordinates, key=lambda item: (item[1][0], item[1][1]))

    # Create a sub-folder based on the image's filename
    image_filename = os.path.basename(image_path)
    image_folder = os.path.splitext(image_filename)[0]
    image_folder_path = os.path.join(output_folder, image_folder)

    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    # Constructing output folder path and saving images
    for i in range(len(sorted_paragraphs)):
        output_name = f"{i+1:03d}.png"
        output_path = os.path.join(image_folder_path, output_name)

        cv2.imwrite(output_path, sorted_paragraphs[i][0])

    # ## Uncomment this part if you want the final output to be displayed (WARNING LAG) ###
    # # Show extracted paragraphs
    # for i in range(len(sorted_paragraphs)):
    #     cv2.imshow('Paragraph', sorted_paragraphs[i][0])
    #     cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    #     cv2.destroyAllWindows()  # Close the window after a key is pressed
    # # _____________________________________________________________________

print("Paragraphs have been extracted and saved to respective folders")
