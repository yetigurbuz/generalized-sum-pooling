import numpy as np

import torch

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import lines as mlines

from scipy.optimize import linprog

from intra_batch_framework.intra_batch.net.gsp.transport_layer import PartialTransportPlan
from intra_batch_framework.intra_batch.net.gsp.pdist_layer import PDistL2


def approx_solver(c, mu, gamma, maxit):

    num_anchors, height, width = c.shape

    c_b = np.expand_dims(c, axis=0).astype(np.float32)


    compute_transport_plan = PartialTransportPlan.apply
    P = compute_transport_plan(torch.tensor(c_b),  mu, gamma, maxit)

    x = P.numpy()[0]

    r = x[0]

    w = 1 / (height * width) - r

    w = (w - w.min()) / (w.max() - w.min())

    return np.expand_dims(w, axis=-1)

def lp_solver(c, mu):
    source_dim = c.shape[1] * c.shape[2]
    target_dim = c.shape[0]

    # constraints
    A = np.concatenate([np.kron(np.ones([1, target_dim+1]), np.eye(source_dim)),
                        np.concatenate([np.zeros(shape=(1, source_dim)),
                                        np.ones(shape=(1, source_dim * target_dim))],
                                       axis=1)],
                       axis=0)

    b1 = np.ones(shape=(source_dim, 1)) / source_dim
    b2 = np.ones(shape=(1, 1)) * mu
    b = np.concatenate([b1, b2], axis=0).flatten()

    # cost
    c0 = np.zeros(shape=(source_dim, 1))
    c1 = c.reshape(-1, 1)

    costs = np.concatenate([c0, c1], axis=0).flatten()

    res = linprog(costs, A_eq=A, b_eq=b)

    x = res['x'].reshape(target_dim + 1, c.shape[1], c.shape[2])

    r = x[0]

    w = 1 / source_dim - r

    w = (w - w.min()) / (w.max() - w.min())

    return np.expand_dims(w, axis=-1)

def create_image():
    width = 10
    height = 10

    base_color = (0., 1., 0.)

    base_image = np.zeros(shape=(width, height, 3)) + np.array(base_color).reshape(1, 1, -1)

    patch_color_1 = (1., 0., 0.)
    patch_color_2 = (0., 0., 1.)

    patch_image_1 = np.zeros(shape=(5, 5, 3)) + np.array(patch_color_1).reshape(1, 1, -1)
    patch_image_2 = np.zeros(shape=(5, 5, 3)) + np.array(patch_color_2).reshape(1, 1, -1)

    p = np.concatenate([np.zeros(shape=(5, 5, 1)) + np.arange(5).reshape(-1, 1, 1),
                        np.zeros(shape=(5, 5, 1)) + np.arange(5).reshape(1, -1, 1)], axis=-1)

    p -= p[2][2].reshape(1, 1, 2)

    d = np.sqrt(np.sum(np.square(p), axis=-1, keepdims=True))

    w = 1. - (d - d.min()) / (d.max() - d.min())

    w = 0.5 + 0.5 * w

    patch_image_1 *= w
    patch_image_2 *= w

    base_image[:5, :5, :] = patch_image_1
    base_image[5:, 5:, :] = patch_image_2

    return base_image

def show_results(img, w_lp, w_hi, w_lo, mu, gamma_hi, gamma_lo, fontsize):


    fig, axarr = plt.subplots(2, 2, constrained_layout=True, figsize=(4, 4.3))

    canvas = FigureCanvas(fig)

    fig.suptitle(r'pooling weights for $\mu={}$'.format(mu), fontsize=fontsize)

    captions = ['feature map',
                r'linear programming',
                r'$(\varepsilon={})$-smoothed'.format(gamma_hi),
                r'$(\varepsilon={})$-smoothed'.format(gamma_lo)]

    cmaps = [None, 'gray', 'gray', 'gray']

    plots = [img, 1-w_lp, 1-w_hi, 1-w_lo]

    for p, ax, caption, cmap in zip(plots, fig.axes, captions, cmaps):
        ax.imshow(p, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(caption, fontsize=fontsize)

    plt.show()

    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    save_image = image.reshape(int(height), int(width), 3)
    return save_image

    '''plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(w_lp), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(np.squeeze(w_hi), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(np.squeeze(w_lo), cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.close()'''


compute_pdists = PDistL2()

img = create_image()

anchors = np.array([[1., 0., 0.], [0., 0., 1.]])

c = np.transpose(compute_pdists(torch.tensor(anchors), torch.tensor(img)).numpy(), (1, 0, 2))

mu = 0.2

gamma_hi = 50
gamma_lo = 0.5

fontsize = 12

w_lp = lp_solver(c, mu)
w_high_gamma = approx_solver(c, mu, gamma=gamma_hi, maxit=1000)
w_low_gamma = approx_solver(c, mu, gamma=gamma_lo, maxit=1000)

save_image = show_results(img, w_lp, w_high_gamma, w_low_gamma,
                          mu, gamma_hi, gamma_lo, fontsize=fontsize)