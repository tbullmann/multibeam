import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft
from tifffile import imread as tifread

from snr import snr_from_corr, snr_from_autocorr
from utils import Hexagon, EXAMPLE_HEXAGON_CORTEX, EXAMPLE_HEXAGON_HIPPOCAMPUS

snr_bins = np.linspace(0, 20, num=21)
snr_ticks = np.linspace(0, 20, num=11)

def make_figure():
    # frank_example()
    frank_comparison()
    # kim_example()
    kim_comparison()


def frank_comparison():

    image_pairs, index_pairs = image_and_index_pairs_hippocampus()
    snr_hippocampus = [snr_from_corr(imageA.ravel(), imageB.ravel()) for imageA, imageB in image_pairs]

    image_pairs, index_pairs = image_and_index_pairs_cortex()
    snr_cortex = [snr_from_corr(imageA.ravel(), imageB.ravel()) for imageA, imageB in image_pairs]

    plt.figure('SNR_Frank_Comparison', figsize=(8, 4))

    ax1 = plt.subplot(121)
    plt.hist(snr_hippocampus, bins=snr_bins)
    plt.title('Marmoset, Hippocampus\n(Distribution for %d pairs)' % len(snr_hippocampus))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0,10))
    plt.xticks(snr_ticks)
    ax1.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_hippocampus),
         horizontalalignment='center',
         verticalalignment='top',
         transform=ax1.transAxes)

    ax2 = plt.subplot(122)
    plt.hist(snr_cortex, bins=snr_bins)
    plt.title('Marmoset, Cortex\n(Distribution for %d pairs)' % len(snr_cortex))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0,10))
    plt.xticks(snr_ticks)
    ax2.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_cortex),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()


def frank_example():
    image_pairs, index_pairs = image_and_index_pairs_cortex()
    # snr = [ snr_from_corr(imageA.ravel(), imageB.ravel()) for imageA, imageB in image_pairs ]
    plt.figure('SNR_Frank_Example_Cortex', figsize=(8, 4), )
    imageA, imageB = image_pairs[0]
    i, j = index_pairs[0]
    ax1 = plt.subplot(121)
    thumbs = Hexagon(EXAMPLE_HEXAGON_CORTEX)
    thumbs.remove_focus_and_beam_artifact()
    thumbs.plot()
    thumbs.plot_tiles((i, j))
    plt.axis('off')
    plt.title('Marmoset, Cortex\n(Tile overlap)')
    ax1 = plt.subplot(185)
    ax1.imshow(imageB, cmap='gray_r')
    plt.title('Tile %d' % j)
    plt.axis('off')
    ax2 = plt.subplot(186)
    ax2.imshow(imageA, cmap='gray_r')
    plt.title('Tile %d' % i)
    plt.axis('off')
    ax3 = plt.subplot(187)
    mean = (imageA + imageB) / 2
    ax3.imshow(mean, cmap='gray_r')
    plt.title('Mean')
    plt.axis('off')
    ax4 = plt.subplot(188)
    diff = (imageA - np.mean(imageA)) - (imageB - np.mean(imageB))
    ax4.imshow(diff, cmap='gray_r')
    plt.title('Difference')
    plt.axis('off')
    plt.show()


def kim_comparison():

    data = Hexagon(EXAMPLE_HEXAGON_HIPPOCAMPUS, prefix='image')
    data.remove_focus_and_beam_artifact()
    snr_hippocampus = [snr_from_autocorr(data.stack[:,:,x]) for x in xrange(data.stack.shape[2])]

    data = Hexagon(EXAMPLE_HEXAGON_CORTEX, prefix='image')
    data.remove_focus_and_beam_artifact()
    snr_cortex = [snr_from_autocorr(data.stack[:,:,x]) for x in xrange(data.stack.shape[2])]

    stack = tifread('data2/snemi2d/train-volume.tif')
    snr_snemi2d = [snr_from_autocorr(stack[x, :, :]) for x in xrange(stack.shape[0])]

    stack = tifread('data2/snemi3d/train-input.tif')
    snr_snemi3d = [snr_from_autocorr(stack[x, :, :]) for x in xrange(stack.shape[0])]

    plt.figure('SNR_Kim_Comparison', figsize=(8, 8))

    ax1 = plt.subplot(221)
    plt.hist(snr_hippocampus, bins=snr_bins)
    plt.title('Marmoset, Hippocampus\n(Distribution for %d images)' % len(snr_hippocampus))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0, 10))
    plt.xticks(snr_ticks)
    ax1.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_hippocampus),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax1.transAxes)

    ax2 = plt.subplot(222)
    plt.hist(snr_cortex, bins=snr_bins)
    plt.title('Marmoset, Cortex\n(Distribution for %d images)' % len(snr_cortex))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0, 10))
    plt.xticks(snr_ticks)
    ax2.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_cortex),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax2.transAxes)

    ax3 = plt.subplot(223)
    plt.hist(snr_snemi2d, bins=snr_bins)
    plt.title('Mouse, Cortex (SNEMI2D)\n(Distribution for %d images)' % len(snr_snemi2d))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0, 10))
    plt.xticks(snr_ticks)
    ax3.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_snemi2d),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax3.transAxes)

    ax4 = plt.subplot(224)
    plt.hist(snr_snemi3d, bins=snr_bins)
    plt.title('Mouse, Cortex (SNEMI3D)\n(Distribution for %d images)' % len(snr_snemi3d))
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.xlim((0, 10))
    plt.ylim((0, 20))
    plt.xticks(snr_ticks)
    ax4.text(0.5, 0.95, 'median = %1.3f' % np.median(snr_snemi3d),
             horizontalalignment='center',
             verticalalignment='top',
             transform=ax4.transAxes)

    plt.tight_layout()
    plt.show()


def kim_example():

    data = Hexagon(EXAMPLE_HEXAGON_CORTEX)
    data.remove_focus_and_beam_artifact()
    beam_index=1
    img = data.stack[:, :, beam_index]

    plt.figure('SNR_Kim_Example_Cortex', figsize=(8, 4), )

    imgFT = fft(img - np.mean(img), axis=1)
    imgAC = ifft(imgFT * np.conjugate(imgFT), axis=1).real
    AC = np.median(imgAC, axis=0)
    mu = AC[len(AC) // 2]

    ax1 = plt.subplot(121)
    ax1.imshow(img, cmap='gray_r')
    plt.title('Tile %d' % beam_index)
    plt.axis('off')

    plt.subplot(122)
    plt.plot(imgAC.T, color='gray')
    plt.plot(AC, marker='o', color='blue', label='median AC', zorder=10)
    plt.plot((0,10),(mu,mu),'b--')
    plt.plot((0,10),(AC[0],AC[0]),'b:')
    plt.annotate(s='', xy=(1, AC[0]), xytext=(1, AC[1]),  arrowprops=dict(edgecolor=None, facecolor='red', shrink=0), ha="center", va="center")
    plt.annotate(s='', xy=(1, AC[1]), xytext=(1, mu),  arrowprops=dict(facecolor='green', shrink=0),)
    plt.text(1, (AC[0] + AC[1])/2, '  noise', va='center', fontsize=14)
    plt.text(1, (mu + AC[1]) / 2, '  signal', va='center', fontsize=14)
    plt.legend()
    plt.xlim(0,10)
    plt.ylim(0,15000)
    plt.xlabel('lag [pixel]')
    plt.xticks(xrange(11))
    plt.ylabel('correlation [AU]')


    plt.tight_layout()
    plt.show()


def image_and_index_pairs_cortex():
    pickle_filename = 'temp/image_and_index_pairs_cortex.p'
    if os.path.isfile(pickle_filename):
        image_pairs, index_pairs = pickle.load(open(pickle_filename,'rb'))
    else:
        data = Hexagon(EXAMPLE_HEXAGON_CORTEX, prefix='image')
        data.remove_focus_and_beam_artifact()
        image_pairs, index_pairs = data.overlaps
        pickle.dump((image_pairs, index_pairs), open(pickle_filename, 'wb'))
    return image_pairs, index_pairs


def image_and_index_pairs_hippocampus():
    pickle_filename = 'temp/image_and_index_pairs_hippocampus.p'
    if os.path.isfile(pickle_filename):
        image_pairs, index_pairs = pickle.load(open(pickle_filename,'rb'))
    else:
        data = Hexagon(EXAMPLE_HEXAGON_HIPPOCAMPUS, prefix='image')
        data.remove_focus_and_beam_artifact()
        image_pairs, index_pairs = data.overlaps
        pickle.dump((image_pairs, index_pairs), open(pickle_filename, 'wb'))
    return image_pairs, index_pairs