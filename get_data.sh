# Download dataset
fileid="10NNtkkQgrEHx4Sa7NbVWoLLgSfVHd6tm"
filename="data/face_mask_dataset.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# Unzip contents
unzip -q ${filename} -d data

# Clean up files
rm ${filename}
rm cookie
