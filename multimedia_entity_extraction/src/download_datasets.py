
from data.downloader import VisualGroundingDatasetDownloader

import os
from config.config import cfg

if __name__ == '__main__':
    # print(cfg)
    downloader = VisualGroundingDatasetDownloader(cfg)
    downloader.download()
