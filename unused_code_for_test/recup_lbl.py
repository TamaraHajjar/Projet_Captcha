# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:38:31 2024

@author: MC
"""

import os

folder_path = 'C:/Users/MC/Desktop/PFE S5/dataset/dread'

for lbl_file in os.listdir(folder_path):
    if lbl_file.endswith('.solve'):  # Check if the file has .solve extension
        old_file = os.path.join(folder_path, lbl_file)  # Get the full path of the file
        new_file = os.path.join(folder_path, lbl_file.replace('.solve', '.txt'))  # Replace extension with .txt
        os.rename(old_file, new_file)  # Rename the file
       
file_contents = []

for lbl_file in os.listdir(folder_path):
    if lbl_file.endswith('.txt'):
        file_path = os.path.join(folder_path, lbl_file)
        with open(file_path, 'r') as file:
            content = file.read()  # Read the content of the file
            file_contents.append(content)  # Add it to the list

file_contents    
