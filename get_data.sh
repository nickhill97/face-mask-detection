download_zip_file_from_google_drive()
{
    fileid=$1
    filename=$2

    # Download zip file
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

    echo "Downloaded $filename"

    # Unzip contents
    unzip -q ${filename} -d data

    # Clean up files
    rm ${filename}
    rm cookie

    echo "Download finished"
}

fileid="10NNtkkQgrEHx4Sa7NbVWoLLgSfVHd6tm"
filename="data/face_mask_dataset.zip"
download_zip_file_from_google_drive ${fileid} ${filename}

fileid="1wTpmnuTbA9sh_N8KSaFCUJyeqEGo3OlF"
filename="data/non_mask_faces.zip"
download_zip_file_from_google_drive ${fileid} ${filename}
