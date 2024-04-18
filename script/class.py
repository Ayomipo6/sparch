import os
import shutil

class LabelProcessor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.labels = set()  # Using a set to store unique labels

    def process_folder(self):
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                filename = os.path.join(root, file)
                label = self.extract_label(filename)
                self.labels.add(label)
                self.move_file(filename, label)

    def extract_label(self, filename):
        # Extract digit from the filename
        parts = filename.split("-")
        digit = parts[-1].split(".")[0].split("_")[-1]

        # Adjust digit based on language
        if "german" in filename.lower():
            digit = str(int(digit) + 10)

        return digit

    def move_file(self, filename, label):
        label_folder = os.path.join(self.data_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        destination = os.path.join(label_folder, os.path.basename(filename))
        shutil.move(filename, destination)


if __name__ == "__main__":
    data_folder = "/Users/demiladepopoola/Development/hd_spiketrum_encoded/"
    processor = LabelProcessor(data_folder)
    processor.process_folder()
