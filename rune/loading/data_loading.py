import shutil
import urllib3
import certifi
import os
import re
from tqdm import tqdm
from .preprocessing import preprocess

class Loader10x:


    def __init__(self, folder='data/raw/'):

        self.http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                        ca_certs=certifi.where(),
                                        strict=False)
        self.donors = [f'donor{i}' for i in range(1,5)]
        self.folder = folder

        with open('data/links.txt', 'r+') as f:
            self.apilist = f.readlines()

    def download_dataset(self):

        folder = self.folder

        if not os.path.isdir(folder):
            os.mkdir(folder)

        for api in tqdm(self.apilist):
            for donor in self.donors:
                get_url = re.sub('donor1', donor, api)
                get_url = get_url.strip()
                filename = os.path.join(folder, get_url.split('/')[-1])
                
                if not os.path.isfile(filename):
                    self.download_file(get_url, filename)
                

    def preprocess_matrices(self, save_path=None):

        if save_path:
            preprocess(self.folder, save_path)
        else:
            preprocess(self.folder)
    
    def download_file(self, get_url: str, filename: str):
        with self.http.request('GET', get_url, preload_content=False) as res:
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(res, out_file)