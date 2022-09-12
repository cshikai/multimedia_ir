import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap
import cv2
import torch
from PIL import Image
import numpy as np
class BoundingBoxGenerator:

    FOLDER_PATH = '/data/heatmaps/'

    MODE = 'MAX'
    
    def __init__(self):
        if not os.path.exists(self.FOLDER_PATH):
            os.makedirs(self.FOLDER_PATH)

    def generate(self,word_image_heatmap,raw_texts,image_urls,indexes):

        batch_size = word_image_heatmap.shape[0]

        for n in range(batch_size):
            raw_text = raw_texts[n]
            image_url = image_urls[n]
            index = indexes[n]
            image =  mpimg.imread(os.path.join('/data/',image_url))
            words = raw_text.split(' ')
            for i,word in enumerate(words):
                fig, axs = plt.subplots(1, 5, )
                _imgplot = axs[0].imshow(image)

                for l in range(4):
                    heatmap_layer = word_image_heatmap[n,:, :, i, l].numpy()
                    # minimum = np.min(heatmap_layer)
                    # maximum = np.max(heatmap_layer)
                    # scale = max(abs(minimum),abs(maximum))
                    # heatmap_layer = heatmap_layer / scale
                    # unedited_layer = word_image_heatmap[n,:, :, i, l].numpy()
                    heatmap_layer = ((heatmap_layer + 1) *0.5 * 255).astype(np.uint8)
                    heatmap_layer_c = cv2.cvtColor(heatmap_layer, cv2.COLOR_GRAY2BGR)
                    # thresh = cv2.threshold(heatmap_layer, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    thresh = cv2.threshold(heatmap_layer, 0, 5, cv2.THRESH_BINARY )[1]
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        x,y,w,h = cv2.boundingRect(c) #36,255,12, #213,75,240
                        cv2.rectangle(heatmap_layer_c, (x, y), (x + w, y + h), (0,0,255), 1)

                    axs[1+l].imshow(heatmap_layer_c)
                _title = axs[1].set_title( 
                    '\n'.join(wrap(word + ' - ' + raw_text, 60)), size=4)

                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                
                plt.savefig(os.path.join(
                    self.FOLDER_PATH, '{}_{}.png'.format(index, i)), dpi=300)




