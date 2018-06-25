import numpy as np
import skimage.io
import skimage.filters

def _check_image_input(im):
    if not isinstance(im, np.ndarray):
        raise RuntimeError("Need to provide a numpy array, image has type " + str(type(im)))
    if not len(im.shape) == 2:
        raise RuntimeError("Need to provide an array with shape (n, m). Provided array has shape " + str(im.shape))

def _check_function_input(im, thresh_func, args=()):
    
    if not callable(thresh_func):
        raise RuntimeError("The provided function is not callable")
        
    if not isinstance(thresh_func(im, args), float):
        raise RuntimeError("The provided function does not output float, gives " + str(type(thresh_func(im, args))))
        
    return True

def segment(im, thresh_func=skimage.filters.threshold_otsu, thresh_approach='By value', args=()):
    """Function to threshold an image using skimage functions. 
    The user pases the desired function to determine the threshold
    for the data (or a value to use as the threshold).
    
    Parameters
    ----------
    x : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to threshold.
    thresh_func : function
        The function to use to calculate the thresholding. Should
        return a single scalar value.
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        Boolean array with location of thresholded objects.
    """
    
    _check_image_input(im)
    _check_function_input(im, thresh_func, args)
    
    thresh = thresh_func(im, args)
    
    pass