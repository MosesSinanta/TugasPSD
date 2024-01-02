from pydub import AudioSegment

def extract_first_4_seconds(input_path, output_folder, ffmpeg_path=None):
    # Use the provided ffmpeg_path or use the default (system PATH)
    AudioSegment.converter = ffmpeg_path

    audio = AudioSegment.from_file(input_path)

    # Extract the first 4 seconds
    segment = audio[:4000]

    # Form the output file path with the original filename
    filename = os.path.basename(input_path)
    filename_without_extension, extension = os.path.splitext(filename)
    output_file = f"{output_folder}/{filename_without_extension}{extension}"

    # Save the segment
    segment.export(output_file, format="wav")
    print(f"First 4 seconds saved to {output_file}")

def extract_first_4_seconds_all_files(input_folder, output_folder, ffmpeg_path=None):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            extract_first_4_seconds(input_path, output_folder, ffmpeg_path)

# Example usage:
input_folder = "path/to/your/input/folder"
output_folder = "path/to/your/output/folder"

# Specify the path to FFmpeg if it's not in your system PATH
ffmpeg_path = "path/to/ffmpeg"  # Replace with the actual path to FFmpeg executable

extract_first_4_seconds_all_files(input_folder, output_folder, ffmpeg_path)
