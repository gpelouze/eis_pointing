#!/usr/bin/env python3

def map_extent(img, coordinates):
    ''' Compute the extent to use in ax.imshow to plot an image with
    coordinates

    Parameters
    ==========
    img : np.ndarray
        A 2D array.
    coordinates : tuple
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values.

    The computed boundaries are the centers of the corresponding pixel. This
    differs from the behaviour of ax.imshow with the extent keyword, where the
    boundaries are the left, right, top, or bottom of the bounding pixels.
    '''
    ny, nx = img.shape
    try:
        xmin, xmax, ymin, ymax = coordinates
    except ValueError:
        x, y = coordinates
        xmin = x[0];  xmax = x[-1]
        ymin = y[0];  ymax = y[-1]
        # swap values if values were in decreasing order
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    xmin -= dx / 2;  xmax += dx / 2
    ymin -= dy / 2;  ymax += dy / 2
    return xmin, xmax, ymin, ymax

def plot_map(ax, img, coordinates=None, **kwargs):
    ''' Plot an image with coordinates

    Parameters
    ==========
    ax : matplotlib.axes.Axes
    img : np.ndarray
        A 2D array.
    coordinates : tuple or None (default: None)
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values. If None, this function
        is equivalent to ax.imshow().
    **kwargs :
        Passed to ax.imshow.

    This function relies on map_extent(), see its docstring.
    '''

    if coordinates:
        extent = map_extent(img, coordinates)
    else:
        try:
            extent = kwargs.pop('extent')
        except KeyError:
            extent = None

    im = ax.imshow(
        img,
        extent=extent,
        **kwargs)

    return im
