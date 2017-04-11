import numpy as np
from matplotlib import gridspec as gs
from matplotlib import pyplot as plt
from skimage.feature import register_translation

from utils import Hexagon, EXAMPLE_HEXAGON_HIPPOCAMPUS


def get_shift_by_afterimage():
    """Use afterimages produced by previous imaging the adjacent hexagons"""
    data = Hexagon(EXAMPLE_HEXAGON_HIPPOCAMPUS, prefix='thumbnail')
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

    data = Hexagon(EXAMPLE_HEXAGON_HIPPOCAMPUS, prefix='thumbnail')
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
    get_shift_by_afterimage()
    get_shift_by_phase_correlation()

