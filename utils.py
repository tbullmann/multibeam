import os
import pandas as pd
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from pyramid.decorator import reify

EXAMPLE_HEXAGON = 'data/000046'


class ImageIterator:
    def __init__(self, path, prefix='thumbnail'):
        """
        :param path: base path containing a file with coordinates and image filenames as well as images
        :param prefix:  'thumbnail' : display thumbnails
                        'image': display images
        """
        self.path = path
        textfilename = os.path.join(path, prefix + '_coordinates.txt')
        self.tiles = pd.read_table(textfilename, names=['filename', 'x', 'y', 'z'])
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
        :param alpha: transparancy; default: 1 (not transparent)
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
            plt.text(tile.x+width//2, tile.y+height//2, tile.name)

    @reify
    def stack(self):
        """Loading all images into a stack"""
        image_list = []
        for raw_image, tile in self:
            image_list.append(raw_image)
        return np.stack(image_list, axis=2)

    @reify
    def vignette(self):
        return np.median(self.stack, axis=2)

    @reify
    def offsets(self):
        return np.median(self.stack, axis=(0, 1))


class ImageStack(ImageIterator):

    def remove_focus_and_beam_artifact(self):
        """
        The 'focus artifact' produces the same vignette in all tiles, whereas the 'beam artifact' (maybe an
        'PMT artifact') is an offset for each tile.
        """
        self.stack = self.stack \
                     - self.vignette[:, :, None] + 127 \
                     - self.offsets[None, None, :] + 127
        self.images = self.tiles.iterrows()   # Start new
        self.next = self.next_from_stack   # Use the stack for plotting

    def next_from_stack(self):
        index, tile = self.images.next()
        if tile is None:
            raise StopIteration()
        else:
            raw_image = self.stack[:, :, index]
            return raw_image, tile


def show_dataset():
    ImageIterator(EXAMPLE_HEXAGON).plot()
    # ImageIterator('data/000046', prefix='image').plot()  # slow
    plt.autoscale()
    plt.show()


def show_dataset_cleaned():
    data = ImageStack(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()
    plt.imshow(data.vignette)
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
