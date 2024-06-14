import gdown
from pathlib import Path


def downloader(output_file):
    
    file_id = "17nxZ4DVu5yB9Mof-mIpvr24gn9dDwwWH" # not ignore criterion
    prefix = 'https://drive.google.com/uc?/export=download&id='

    url_download = prefix+file_id
    
    if not Path(output_file).exists():
        print("Downloading...")
        gdown.download(url_download, output_file)
        print("Download Finish...")