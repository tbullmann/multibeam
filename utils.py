import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyramid.decorator import reify
from scipy.misc import imread
from scipy.stats import threshold, pearsonr
from skimage.feature import register_translation
from numpy.fft import fft, ifft

EXAMPLE_HEXAGON = 'data/000046'
EXAMPLE_HEXAGON = 'data/000007'


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
            plt.text(tile.x+width//2, tile.y+height//2, tile.beam_index)

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
        overlaps, n = pairwise_overlaps(height, width, x, y)
        assert any(n < 7)  # every tile has less than 7 neighbors
        assert overlaps.shape[0] == (37*6 + 6*3*4 + 6*3)/2  # 37 inner tile, 6*3 on the edges, 6 on the corners with 6, 4, 3 neightbors, respectively

        def adjust_overlap(overlap):  # Adjust overlap by phase correlation
            imageA, imageB = overlapping_parts_of_image_pair(round(overlap.dx), round(overlap.dy),
                                              self.stack[:, :, overlap.i], self.stack[:, :, overlap.j])
            shift, _, _ = register_translation(imageB, imageA, upsample_factor=1)
            ddy, ddx = shift
            return pd.Series({'dy': round(ddy), 'dx': round(ddx)})

        adjusted = overlaps.apply(adjust_overlap, axis=1)
        overlaps.dy += adjusted.dy
        overlaps.dx += adjusted.dx

        print overlaps

        return overlaps


def pairwise_overlaps(height, width, x, y):
    """
    :param height, width: size of a single image
    :param x, y: coordinates of the images
    :return: panda DataFrame with
        i, j: pairwise indices of the images with non zero overlap
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

    overlaps = pd.DataFrame({'i': i, 'j': j, 'dx': dx, 'dy': dy, 'A': A}).sort_values(by='A', ascending=False)

    return overlaps, n


def overlapping_parts_of_image_pair(dx, dy, imageA, imageB):
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


def snr_from_corr(x, y):
    """
    Estimate signal to noise ratio from two different versions of the same signal contaminated by uncorrelated noise,
    using the sample cross-correlation coefficient (Pearson correlation coefficient).
    Note: Frank and Al-Ali, 1975, Nature
    :param x, y: 1D arrays
    :return: snr: signal to noise ratio
    """
    N = len(x)
    r_N, _ = pearsonr(x, y)
    if N > 10000:
        snr = r_N / (1 - r_N)
    else:
        snr = np.exp(-2/(N-3))*(r_N/(1-r_N)+1/2)-1/2
    return snr


def add_snr_from_overlap(data):

    def snr_from_overlap(overlap):
        imageA = data.stack[:, :, overlap.i]
        imageB = data.stack[:, :, overlap.j]

        dx = round(overlap.dx)
        dy = round(overlap.dy)
        # print("Given translation: dy = %1.1f, dx = %1.1f)" % (dy, dx))

        imageA, imageB = overlapping_parts_of_image_pair(dx, dy,
                                                         imageA, imageB)
        snr = snr_from_corr(imageA.ravel(), imageB.ravel())
        return snr

    data.overlaps['snr'] = data.overlaps.apply(snr_from_overlap, axis=1)


def show_dataset():
    """Show single hexagon both as thumbnails and (full resolution) image"""
    Hexagon(EXAMPLE_HEXAGON).plot()
    # Hexagon(EXAMPLE_HEXAGON, prefix='image').plot()  # slow
    plt.autoscale()
    plt.axis('off')
    plt.show()


def show_dataset_cleaned():
    """Show how to clean up the images"""
    data = Hexagon(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()
    plt.imshow(data.vignette, cmap='gray_r')
    plt.show()
    data.plot()
    plt.autoscale()
    plt.show()


def snr_by_frank():
    """"""
    data = Hexagon(EXAMPLE_HEXAGON, prefix='image')
    data.remove_focus_and_beam_artifact()
    add_snr_from_overlap(data)

    print data.overlaps.snr

    ax1 = plt.subplot(121)
    plt.hist(data.overlaps.snr, bins=50)
    plt.title('distribution for %d pairs of tiles' % data.overlaps.shape[0])
    plt.xlabel('snr')
    plt.ylabel('count')

    ax2 = plt.subplot(122)
    plt.hist(np.log(data.overlaps.snr), bins=50)
    plt.title('distribution for %d pairs of tiles' % data.overlaps.shape[0])
    plt.xlabel('log snr')
    plt.ylabel('count')

    plt.show()

    for _, overlap in data.overlaps.iterrows():

        imageA, imageB = overlapping_parts_of_image_pair(round(overlap.dx), round(overlap.dy),
                                          data.stack[:, :, overlap.i], data.stack[:, :, overlap.j])
        ax1 = plt.subplot(141)
        ax1.imshow(imageA, cmap='gray_r')
        plt.title('Tile %d' % overlap.i)
        plt.axis('off')

        ax2 = plt.subplot(142)
        ax2.imshow(imageB, cmap='gray_r')
        plt.title('Tile %d' % overlap.j)
        plt.axis('off')

        ax3 = plt.subplot(143)
        mean = (imageA + imageB) / 2
        ax3.imshow(mean, cmap='gray_r')
        plt.title('Mean')
        plt.axis('off')

        ax4 = plt.subplot(144)
        diff = (imageA - np.mean(imageA)) - (imageB - np.mean(imageB))
        ax4.imshow(diff, cmap='gray_r')
        plt.title('Difference')
        plt.axis('off')

        plt.show()

def snr_by_kim():

    data = Hexagon(EXAMPLE_HEXAGON, prefix='thumbnail')
    data.remove_focus_and_beam_artifact()

    data.tiles['snr'] = data.tiles.beam_index.apply(lambda x: snr_from_autocorr(data.stack[:,:,x]))
    print data.tiles.snr

    ax1 = plt.subplot(121)
    plt.hist(data.tiles.snr, bins=25)
    plt.title('distribution for %d tiles' % data.tiles.shape[0])
    plt.xlabel('snr')
    plt.ylabel('count')

    ax2 = plt.subplot(122)
    plt.hist(np.log(data.tiles.snr), bins=25)
    plt.title('distribution for %d tiles' % data.tiles.shape[0])
    plt.xlabel('log snr')
    plt.ylabel('count')

    plt.show()

    for beam_index in data.tiles.beam_index:
        img = data.stack[:,:,beam_index]

        imgFT = fft(img - np.mean(img), axis=1)
        imgAC = ifft(imgFT * np.conjugate(imgFT), axis=1).real
        AC = np.median(imgAC, axis=0)

        ax1 = plt.subplot(121)
        ax1.imshow(img, cmap='gray_r')
        plt.title('Tile %d' % beam_index)
        plt.axis('off')

        plt.subplot(143)
        plt.plot(imgAC.T, color='gray')
        plt.plot(AC, color='blue', label='median AC')
        plt.legend()
        plt.xlim(0,10)
        plt.xlabel('lag')

        plt.show()


def snr_from_autocorr(img):
    imgFT = fft(img - np.mean(img), axis=1)
    imgAC = ifft(imgFT * np.conjugate(imgFT), axis=1).real
    AC = np.median(imgAC, axis=0)
    snr = (AC[1] - AC[len(AC)//2]) / (AC[0] - AC[1])
    return snr


if __name__ == "__main__":
    # show_dataset()
    # show_dataset_cleaned()
    # snr_by_frank()
    snr_by_kim()
