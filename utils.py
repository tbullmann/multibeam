import os
import re
import pandas as pd
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from pyramid.decorator import reify

EXAMPLE_HEXAGON = 'data/000046'


class ImageIterator:
    def __init__(self, path, prefix='thumbnail'):
        """
        :param path: base path containing a file with coordinates and image filenames as well as images
        :param prefix:  'thumbnail' : display thumbnails
                        'image': display images
        :return: self.tiles: dataframe with:
            'filename': filename of the tile (image)
            'x', 'y', 'z': coordinates of the lower left corner of the tiles
            'beam_index': counterclockwise starting from the center (= third number in the filename - 1)
            'file_index': order the tiles are listed in the coordinate file and read into the stack, starts from 0
        """
        self.path = path
        textfilename = os.path.join(path, prefix + '_coordinates.txt')
        self.tiles = pd.read_table(textfilename, names=['filename', 'x', 'y', 'z'])
        self.tiles['beam_index'] = self.tiles['filename'].map(lambda x: int(re.findall(r'\d+', x)[2])-1)
        self.tiles['file_index'] = self.tiles.index
        self.images = self.tiles.iterrows()

    def __iter__(self):
        return self

    def next(self):
        index, tile = self.images.next()
        if tile is None:
            raise StopIteration()
        else:
            fullfilename = os.path.join(self.path, tile.filename)
            raw_image = imread(fullfilename, mode='L')
            return raw_image, tile

    def plot(self, transform=None, cmap='gray_r', alpha=1):
        """
        Plotting thumbnails.
        :param transform: image transformation; default None as well
        :param cmap: colormap; default: inverted gray scale
        :param alpha: transparency; default: 1 (not transparent)
        """
        for raw_image, tile in self:
            image = transform(raw_image) if transform else raw_image
            height, width = image.shape[0:2]
            plt.imshow(image,
                       vmin=0,
                       vmax=255,
                       cmap=cmap,
                       alpha=alpha,
                       extent=[tile.x, tile.x+width, tile.y+height, tile.y])
            plt.text(tile.x+width//2, tile.y+height//2, tile.beam_index)

    @reify
    def stack(self):
        """Loading all images into a stack, sorted according to the beam_index"""
        image_list = []
        for raw_image, tile in self:
            image_list.append(raw_image)
        return np.stack(image_list, axis=2)[:, :, np.argsort(self.tiles['beam_index'])]

    @reify
    def vignette(self):
        return np.median(self.stack[:, :, 0:37], axis=2)

    @reify
    def offsets(self):
        return np.median(self.stack, axis=(0, 1))


class ImageStack(ImageIterator):

    def remove_focus_and_beam_artifact(self):
        """
        The 'focus artifact' produces the same vignette in all tiles, whereas the 'beam artifact' (maybe an
        'PMT artifact') is an offset for each tile.
        """
        self.stack = self.stack - self.offsets[None, None, :] + 127
        self.stack = self.stack - self.vignette[:, :, None] + 127
        self.images = self.tiles.iterrows()   # Start new
        self.next = self.next_from_stack   # Use the stack for plotting

    def next_from_stack(self):
        index, tile = self.images.next()
        if tile is None:
            raise StopIteration()
        else:
            raw_image = self.stack[:, :, tile.beam_index]
            return raw_image, tile


def show_dataset():
    ImageIterator(EXAMPLE_HEXAGON).plot()
    # ImageIterator('data/000046', prefix='image').plot()  # slow
    plt.autoscale()
    plt.axis('off')
    plt.show()


def show_dataset_cleaned():
    data = ImageStack(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()
    plt.imshow(data.vignette, cmap='gray_r')
    plt.show()
    data.plot()
    plt.autoscale()
    plt.show()


def get_overlap():
    """Use afterimages produced by previous imaging the adjacent hexagons"""
    pass

if __name__ == "__main__":
    show_dataset()
    show_dataset_cleaned()
    get_overlap()