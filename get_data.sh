# Download dataset
curl "https://data-flair.s3.ap-south-1.amazonaws.com/Data-Science-Data/face-mask-dataset.zip" --output data/face-mask-dataset.zip

# Unzip contents
unzip -q data/face-mask-dataset.zip -d data
unzip -q data/Dataset/train.zip -d data
unzip -q data/Dataset/test.zip -d data

# Clean up files
rm -rf data/Dataset
rm data/face-mask-dataset.zip

