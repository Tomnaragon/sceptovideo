import numpy as np
import skimage.io
import skimage.filters

def _check_image_input(im):
    if not isinstance(im, np.ndarray):
        raise RuntimeError("Need to provide a numpy array, image has type " + str(type(im)))
    if not len(im.shape) == 2:
        raise RuntimeError("Need to provide an array with shape (n, m). Provided array has shape " + str(im.shape))

def _check_numeric_types(number):
    if (isinstance(number, float)       or
        isinstance(number, int)         or
        isinstance(number, np.int)      or
        isinstance(number, np.int32)    or
        isinstance(number, np.int64)    or
        isinstance(number, np.float)    or
        isinstance(number, np.float16)  or
        isinstance(number, np.float32)  or
        isinstance(number, np.float64)  or
        isinstance(number, np.float128) ):
        return True
    return False
        
def _check_function_input(im, thresh_func, args):
    
    if not callable(thresh_func):
        raise RuntimeError("The provided function is not callable")
        
    func_out = thresh_func(im, *args)
    if not (_check_numeric_types(func_out) or isinstance(func_out, np.ndarray)):
        raise RuntimeError("The provided function must output a numeric or array \
                           provided function returns type " + str(type(func_out)))
    if isinstance(func_out, np.ndarray) and not func_out.shape == im.shape:
        raise RuntimeError("Array output of the function must have same shape as the image \
                           the output array has shape " + str(func_out.shape) +
                           ", image has shape " + str(im.shape))

    return True

def _check_ims_same_dim(im1, im2):
    if not (im1.shape == im2.shape):
        raise RuntimeError("The provided images have different dimension \
        im1: "+str(im1.shape)+", im2: "+str(im2.shape))

def bg_subtract(im1, im2):
    """Function to perform background subtraction on an image
    using a blank image of the arena of interest.
    
    Parameters
    ----------
    im1 : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to subtract the
        background from.
    im2 : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The background image (with only one color chanel) to subtract the
        from im1.
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        image with background subtracted, i.e. im1-im2.
    """
    
    _check_image_input(im1)
    _check_image_input(im2)
    _check_ims_same_dim(im1, im2)
    
    im1 = normalize_convert_im(im1)
    im2 = normalize_convert_im(im2)
    
    im_no_bg = im1-im2
    im_no_bg = normalize_convert_im(im_no_bg)
    return (im_no_bg)

def normalize_convert_im(im):
    im = skimage.img_as_float(im)
    im_norm = im - im.min()
    im_norm = im / im.max()
    return im

def segment(im, thresh_func=skimage.filters.threshold_otsu, args=()):
    """Function to threshold an image using skimage functions. 
    The user pases the desired function to determine the threshold
    for the data (or a value to use as the threshold). This value
    for the threshold level can be a float/int or an array of same shape
    as the input image.
    
    Parameters
    ----------
    x : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to threshold.
    thresh_func : function
        The function to use to calculate the thresholding. Should
        return a single scalar value or a numpy array.
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        Boolean array with location of thresholded objects.
    """
    
    _check_image_input(im)
    _check_function_input(im, thresh_func, args)
    
    thresh = thresh_func(im, *args)
    
    im_bw = im < thresh
    
    # Label binary image; background kwarg says value in im_bw to be background
    im_labeled, n_labels = skimage.measure.label(im_bw, background=0, return_num=True)
    
    return im_bw, im_labeled, n_labels
    