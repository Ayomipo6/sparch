import os
import librosa

def find_max_duration_and_file(data_folder):
    """
    Finds the maximum duration among all audio files in the specified folder and its subfolders,
    and returns both the maximum duration and the filename of the longest audio file.

    Parameters:
    - data_folder: Path to the root folder containing the audio files.

    Returns:
    - A tuple containing the maximum duration (in seconds) and the filename of the audio file
      with the maximum duration.
    """
    max_duration = 0
    max_duration_file = ''

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.flac', '.mp3')):
                filepath = os.path.join(root, file)
                duration = librosa.get_duration(filename=filepath)
                if duration > max_duration:
                    max_duration = duration
                    max_duration_file = filepath

    return max_duration, max_duration_file

# Example usage
data_folder = ''
max_time, longest_file = find_max_duration_and_file(data_folder)
print(f"The maximum duration in the dataset is: {max_time} seconds, found in file: {longest_file}")


