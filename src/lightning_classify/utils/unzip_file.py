import zipfile


def unzip(zip_file_path, to_save):

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(to_save)
    
    