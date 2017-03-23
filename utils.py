import os
import pandas as pd
from matplotlib import pyplot as plt


class ImageIterator:
    def __init__(self, path, prefix='thumbnail'):
        """
        :param path: base path containing a file with coordinates and image filenames as well as images
        :param prefix:  'thumbnail' : display thumbnails
                        'image': display images
        """
        self.path = path
        textfilename = os.path.join(path, prefix + '_coordinates.txt')
        tiles = pd.read_table(textfilename, names=['filename', 'x', 'y', 'z'])
        self.images = tiles.iterrows()

    def __iter__(self):
        return self

    def next(self):
        index, tile = self.images.next()
        if tile is None:
            raise StopIteration()
        else:
            fullfilename = os.path.join(self.path, tile.filename)
            raw_image = plt.imread(fullfilename)
            return raw_image, tile

    def plot(self, transform=None, cmap='gray_r', alpha=1):
        """
        :param transform: image transformation; default sqrt, could be None as well
        :param cmap: colormap; default: inverted gray scale
        :param alpha: transparancy; default: 1 (not transparent)
        :return: ?
        """
        for raw_image, tile in self:
            image = transform(raw_image) if transform else raw_image
            height, width, channels = image.shape
            plt.imshow(image,
                       cmap=cmap,
                       alpha=alpha,
                       extent=[tile.x, tile.x+width, tile.y+height, tile.y])
            plt.text(tile.x+width//2, tile.y+height//2, tile.name)


def load_dataset():
    ImageIterator('data/000046').plot()
    plt.autoscale()
    plt.show()


if __name__ == "__main__":
    load_dataset()
