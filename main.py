from files import video_processing,analysis_llm
# Step 1: Process the videos and extract frames
video_processing

# Step 2: Generate summaries and flag crime-related content
folder_path = "outputs/matched_frames"
overall_summary, crime_flag = analysis_llm.generate_img_summaries(folder_path)

# Print the overall summary and crime flag
print(f"Overall Summary:\n{overall_summary}\n")

if crime_flag:
    print("Alert: Crime-related activity detected.")
else:
    print("No crime-related activity detected.")