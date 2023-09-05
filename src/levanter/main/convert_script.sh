#!/bin/bash

# Loop through each line in the folder_list.txt file
while IFS= read -r line; do
    # Split the line into folder_name and subfolder_name using comma as delimiter
    IFS="," read -r folder_name step_num <<< "$line"
    
    # Create a destination path
    destination_folder="/jagupard31/scr1/kathli/checkpoints/$folder_name/step-$step_num"
    
    # Download the subfolder using gsutil
    echo "converting $folder_name"
    python /sailhome/kathli/repos/levanter-midi/src/levanter/main/export_lm_to_hf.py --checkpoint_path "$destination_folder" --output_dir "$destination_folder/hf" --config_path "/sailhome/kathli/repos/levanter-midi/config/gpt2_small_export.yaml"
done < /sailhome/kathli/folder_list.txt

wait
