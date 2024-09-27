import os
import cv2


def extract_frames_from_video(video_path, output_dir, frame_size=(128, 128)):
    """
    Extract and preprocess frames from a video file.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save extracted frames.
        frame_size (tuple): Size to resize each frame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        # Resize frame
        resized_frame = cv2.resize(frame, frame_size)

        # Save the frame as an image file
        frame_filename = f"frame_{frame_count}.jpg"
        frame_output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_output_path, resized_frame)

        frame_count += 1
        success, frame = cap.read()  # Read next frame

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")


def process_all_videos_in_folders(main_dir, output_dir_base):
    """
    Processes all videos inside folders with numeric names and extracts frames.
    Args:
        main_dir (str): Directory containing the numeric folders.
        output_dir_base (str): Base directory where extracted frames will be saved.
    """
    # Loop through each folder in the main directory
    for folder_name in os.listdir(main_dir):
        folder_path = os.path.join(main_dir, folder_name)

        if os.path.isdir(folder_path) and folder_name.isdigit():  # Check if folder name is numeric
            print(f"Processing folder: {folder_name}")

            # Loop through each video file in the numeric folder
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)

                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):  # Check if it's a video file
                    video_type = "high" if "high" in video_file.lower() else "low"

                    # Output directory for frames (subfolder based on video type)
                    output_dir = os.path.join(output_dir_base, folder_name, video_type)

                    # Extract frames from the video
                    extract_frames_from_video(video_path, output_dir)



main_directory = './'  # Folder containing numeric subdirectories
output_frames_base = './extracted_frames/'

# Process all videos and extract frames
process_all_videos_in_folders(main_directory, output_frames_base)
