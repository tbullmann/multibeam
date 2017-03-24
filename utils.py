import os
import re
import pandas as pd
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from skimage.feature import register_translation

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
            raw_image = imread(fullfilename, mode='L')   # Force gray scale
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
        """Median of center tiles only"""
        return np.median(self.stack[:, :, 0:37], axis=2)

    @reify
    def offsets(self):
        """Median of each tile"""
        return np.median(self.stack, axis=(0, 1))

    def coordinates(self, beam_index):
        tile = self.tiles.loc[self.tiles['beam_index'] == beam_index]
        return np.hstack([tile.y.values, tile.x.values])


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
    """Show single hexagon both as thumbnails and (full resolution) image"""
    ImageIterator(EXAMPLE_HEXAGON).plot()
    # ImageIterator('data/000046', prefix='image').plot()  # slow
    plt.autoscale()
    plt.axis('off')
    plt.show()


def show_dataset_cleaned():
    """Show how to clean up the images"""
    data = ImageStack(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()
    plt.imshow(data.vignette, cmap='gray_r')
    plt.show()
    data.plot()
    plt.autoscale()
    plt.show()


def get_shift_by_afterimage():
    """Use afterimages produced by previous imaging the adjacent hexagons"""
    data = ImageStack(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()

    sub_stack = data.stack[:, :, (46, 47, 48) ]  #  tiles from the upper left diagonal
    image = np.median(sub_stack, axis=2)

    grid = gs.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
    ax = plt.subplot(grid[0,1])
    axl = plt.subplot(grid[0,0], sharey=ax)
    axb = plt.subplot(grid[1,1], sharex=ax)

    axl.plot(np.mean(image, axis=1), xrange(image.shape[0]))
    axb.plot(np.mean(image, axis=0))
    ax.imshow(image, cmap='gray_r')
    ax.axis('off')
    plt.show()

    # TODO Extract shift from edges, e.g. in the marginal


def get_shift_by_phase_correlation(upsample_factor=1, alpha=0.5, beam_index_A = 0, beam_index_B = 1):
    """Use phase correlation to identify the shift."""

    # # Modified from: http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    # from skimage import data
    # from scipy.ndimage import fourier_shift
    # image = data.camera()
    # shift = (-22.4, 13.32)
    # # The shift corresponds to the pixel offset relative to the reference image
    # offset_image = fourier_shift(np.fft.fftn(image), shift)
    # offset_image = np.fft.ifftn(offset_image).real

    data = ImageStack(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()
    image = data.stack[:, :, beam_index_A]
    offset_image = data.stack[:, :, beam_index_B]

    known_shift = data.coordinates(beam_index_B) - data.coordinates(beam_index_A)
    print("Known offset (y, x): {}".format(known_shift))

    shift, error, diffphase = register_translation(image, offset_image, upsample_factor=upsample_factor)
    print("Estimated offset (y, x): {}".format(shift))

    dy, dx = shift
    # dy, dx = known_shift

    height, width = image.shape[0:2]
    plt.imshow(image, vmin=0, vmax=255, cmap='Blues_r', alpha=alpha,
               extent=[0, width, height, 0])
    plt.imshow(offset_image, vmin=0, vmax=255, cmap='Reds_r', alpha=alpha,
               extent=[dx, dx+width, dy+height, dy])

    plt.autoscale()
    plt.show()

    # TODO Test with images of adjacent hexagons (having a larger overlap, because: FAIL with small overlap)


if __name__ == "__main__":
    show_dataset()
    show_dataset_cleaned()
    get_shift_by_afterimage()
    get_shift_by_phase_correlation()