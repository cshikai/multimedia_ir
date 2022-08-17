import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap
import torch

class WordHeatmapGenerator:

    FOLDER_PATH = '/data/heatmaps/'


    def __init__(self):
        if not os.path.exists(self.FOLDER_PATH):
            os.makedirs(self.FOLDER_PATH)

    def generate(self,word_image_heatmap,raw_texts,image_urls,indexes):

        N = word_image_heatmap.shape[0]


        for n in range(N):
            raw_text = raw_texts[n]
            image_url = image_urls[n]
            index = indexes[n]
            image =  mpimg.imread(os.path.join('/data/',image_url))
            words = raw_text.split(' ')
            for i,word in enumerate(words):
                fig, axs = plt.subplots(1, 5, )
                _imgplot = axs[0].imshow(image)
                for l in range(4):
                    axs[1+l].imshow(word_image_heatmap[n,:, :, i, l].numpy(), cmap="YlGnBu")

                _title = axs[1].set_title( 
                    '\n'.join(wrap(word + ' - ' + raw_text, 60)), size=4)

                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                
                plt.savefig(os.path.join(
                    self.FOLDER_PATH, '{}_{}.png'.format(index, i)), dpi=300)




