import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from pyramid.decorator import reify
from scipy.misc import imread
from scipy.stats import threshold
from skimage.feature import register_translation

EXAMPLE_HEXAGON_HIPPOCAMPUS = 'data/000046'
EXAMPLE_HEXAGON_CORTEX = 'data/000007'


class Hexagon:
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
        self.tiles = self.tiles.sort_values(by='beam_index')

    @reify
    def stack(self):
        """Loading all images into a stack, sorted according to the beam_index"""
        image_list = []
        for _, tile in self.tiles.iterrows():
            fullfilename = os.path.join(self.path, tile.filename)
            raw_image = imread(fullfilename, mode='L')  # Force gray scale
            image_list.append(raw_image)
        return np.stack(image_list, axis=2)

    def plot(self, transform=None, cmap='gray_r', alpha=1):
        """
        Plotting thumbnails.
        :param transform: image transformation; default None as well
        :param cmap: colormap; default: inverted gray scale
        :param alpha: transparency; default: 1 (not transparent)
        """
        for _, tile in self.tiles.iterrows():
            raw_image = self.stack[:, :, tile.beam_index]
            image = transform(raw_image) if transform else raw_image
            height, width = image.shape[0:2]
            plt.imshow(image,
                       vmin=0,
                       vmax=255,
                       cmap=cmap,
                       alpha=alpha,
                       extent=[tile.x, tile.x+width, tile.y+height, tile.y])
        plt.autoscale()

    def plot_tiles(self, beam_indices=range(0, 62)):
        height, width, _ = self.stack.shape
        ax = plt.gca()
        colors = iter(cm.rainbow(np.linspace(0, 1, len(beam_indices))))
        for _, tile in self.tiles.iterrows():
            if tile.beam_index in beam_indices:
                color = next(colors)
                ax.add_patch(patches.Rectangle((tile.x, tile.y), width, height, color=color, fill=False))
                plt.text(tile.x + width // 2, tile.y + height // 2, tile.beam_index, color=color,
                         horizontalalignment='center', verticalalignment='center')
        plt.autoscale()

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

    def remove_focus_and_beam_artifact(self):
        """
        The 'focus artifact' produces the same vignette in all tiles, whereas the 'beam artifact' (maybe an
        'PMT artifact') is an offset for each tile.
        """
        self.stack = self.stack - self.offsets[None, None, :] + 127
        self.stack = self.stack - self.vignette[:, :, None] + 127
        self.images = self.tiles.iterrows()   # Start new

    @reify
    def overlaps(self):
        height, width, _ = self.stack.shape
        x = self.tiles.x
        y = self.tiles.y
        registrations, n = pairwise_registrations(height, width, x, y)
        assert any(n < 7)  # every tile has less than 7 neighbors
        assert registrations.shape[0] == (37*6 + 6*3*4 + 6*3)/2  # 37 inner tile, 6*3 on the edges, 6 on the corners with 6, 4, 3 neightbors, respectively

        def adjust(registration):  # Adjust overlap by phase correlation
            imageA, imageB = overlap_from_registration(int(round(registration.dx)), int(round(registration.dy)),
                                                       self.stack[:, :, int(registration.i)],
                                                       self.stack[:, :, int(registration.j)])
            shift, _, _ = register_translation(imageB, imageA, upsample_factor=1)
            ddy, ddx = shift
            return pd.Series({'dy': int(round(ddy)), 'dx': int(round(ddx))})

        adjusted = registrations.apply(adjust, axis=1)

        registrations.dy += adjusted.dy
        registrations.dx += adjusted.dx

        image_pairs = []
        index_pairs = []
        for registration in registrations.itertuples():
            imageA = self.stack[:, :, registration.i]
            imageB = self.stack[:, :, registration.j]
            dx = registration.dx
            dy = registration.dy
            image_pairs.append(overlap_from_registration(dx, dy, imageA, imageB))
            index_pairs.append((registration.i, registration.j))

        return image_pairs, index_pairs


# static functions for image registration

def pairwise_registrations(height, width, x, y):
    """
    :param height, width: size of a single image
    :param x, y: coordinates of the images
    :return: panda DataFrame with
        i, j: pairwise indices of the images with non zero overlap
        dx, dy: displacement between the images with image i and j
        A: overlap area for image pair with indices i and j
    """

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]

    ox = threshold(width - abs(dx), 0)
    oy = threshold(height - abs(dy), 0)

    A = ox * oy

    non_zero_overlap = A > 0  # use only nonzero overlap
    np.fill_diagonal(non_zero_overlap, False)  # discard i==j because same overlap with itself

    n = np.sum(non_zero_overlap, axis=0)

    non_zero_overlap &= np.tri(*non_zero_overlap.shape).astype(bool)  # discard i<j because redundancy, A(i,j) = A(j,i)

    i, j = np.where(non_zero_overlap)
    A = np.array([A[index] for index in zip(i, j)])
    dx = np.array([dx[index] for index in zip(i, j)])
    dy = np.array([dy[index] for index in zip(i, j)])

    registrations = pd.DataFrame({'i': i, 'j': j,
                                  'dx': dx.astype(int), 'dy': dy.astype(int),
                                  'A': A}
                                 ).sort_values(by='A', ascending=False)

    return registrations, n


def overlap_from_registration(dx, dy, imageA, imageB):
    """
    :param dx, dy: translation
    :param imageA, imageB: two images
    :return: images parts that overlap
    """

    height, width = imageA.shape[0:2]
    if dx > 0:
        imageA = imageA[:, :width - dx]
        imageB = imageB[:, dx:]
    else:
        imageA = imageA[:, -dx:]
        imageB = imageB[:, :width + dx]
    if dy > 0:
        imageA = imageA[:height - dy, :]
        imageB = imageB[dy:, :]
    else:
        imageA = imageA[-dy:, :]
        imageB = imageB[:height + dy, :]

    return imageA, imageB


def show_dataset():
    """
    Show single hexagon both as thumbnails with tile outline and beam index
    Note: Plotting as full resolution image is slow
    """
    # Plot all tiles on top
    data = Hexagon(EXAMPLE_HEXAGON_CORTEX)  # Hexagon(EXAMPLE_HEXAGON, prefix='image')
    data.remove_focus_and_beam_artifact()
    data.plot()
    data.plot_tiles()
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    show_dataset()
