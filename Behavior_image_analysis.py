import numpy as np
import skimage.io
import skimage.filters
import skimage.morphology
import pandas as pd
import sys
import scipy


from functools import partial
import tqdm
import multiprocessing
import imageio

import time

sys.path.append("behavioral_analysis/")
import segmentation as jwseg

import warnings

import motmot.FlyMovieFormat.FlyMovieFormat as FMF


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
    im1 = normalize_convert_im(im1)
    im2 = normalize_convert_im(im2)

    im_no_bg = im1-im2
    #im_no_bg = normalize_convert_im(im_no_bg)
    return (im_no_bg)

def normalize_convert_im(im):
    """Function to normalize an image and convert it to float type.
    Normalized image is between 0 and 1.0.

    Parameters
    ----------
    im : numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) tp get ROI from.

    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m) of type float and in
    range of 0 to 1.0
    """
    im = skimage.img_as_float(im)
    im_norm = im - im.min()
    im_norm = im_norm / im_norm.max()
    return im_norm

def segment(im, thresh_func=skimage.filters.threshold_otsu, args=()):
    """Function to threshold an image using skimage functions.
    The user passes the desired function to determine the threshold
    for the data (or a value to use as the threshold). This value
    for the threshold level can be a float/int or an array of the same shape
    as the input image.

    Parameters
    ----------
    x : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color channel) to threshold.
    thresh_func : function
        The function to use to calculate the thresholding. Should
        return a single scalar value or a numpy array.
    Returns
    -------
    im_bw : 2d numpy.ndarray with shape (n, m)
        Boolean array with location of thresholded objects.
    im_labeled : a labelefield of iumage
    n_labels : number of identified objects in the labelfield
    """
    thresh = thresh_func(im, *args)
    im_bw = im < thresh

    #Remove small objecs, i.e. insect legs that are recognized as
    #separate from the insect body
    im_bw_big_objs = skimage.morphology.remove_small_objects(im_bw)

    #Label binary image; background kwarg says value in im_bw to be
    #background
    im_labeled, n_labels = skimage.measure.label(im_bw_big_objs,
    background=0, return_num = True)

    return im_bw, im_labeled, n_labels

def region_props_to_tuple(rp):
    """Function to extract the region properties from a regionprops
    object.

    Parameters
    ----------
    rp : skimage.measure._regionprops._RegionProperties, a region properties
        object from which to extract attributes
    Returns
    -------
    attributes : tuple. The regionproperty fields for many properites of a blob
    labels : tuple. The corresponding label for teh values in attributes
    """
    labels = ('area', 'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col', 'bbox_area',
              'centroid_row', 'centroid_col', 'convex_area', 'eccentricity', 'equivalent_diameter',
              'euler_number', 'extent', 'filled_area', 'label', 'local_centroid_row', 'local_centoid_col',
              'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length',
              'orientation', 'perimeter', 'solidity', 'weighted_centroid_row', 'weighted_centroid_col',
              'weighted_local_centoid_row', 'weighted_local_centroid_col')

    bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = rp.bbox
    centroid_row, centroid_col = rp.centroid
    local_centroid_row, local_centoid_col = rp.local_centroid
    weighted_centroid_row, weighted_centroid_col = rp.weighted_centroid
    weighted_local_centoid_row, weighted_local_centroid_col = rp.weighted_local_centroid

    attributes = (rp.area, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col,
                  rp.bbox_area, centroid_row, centroid_col, rp.convex_area,
                  rp.eccentricity, rp.equivalent_diameter, rp.euler_number,
                  rp.extent, rp.filled_area, rp.label, local_centroid_row,
                  local_centoid_col, rp.major_axis_length, rp.max_intensity,
                  rp.mean_intensity, rp.min_intensity, rp.minor_axis_length,
                  rp.orientation, rp.perimeter, rp.solidity,
                  weighted_centroid_row, weighted_centroid_col,
                  weighted_local_centoid_row, weighted_local_centroid_col)

    return attributes, labels

def image_analysis(curr_im, bg_im):
    """Combines functions to perform background subtraction of an image
    threshold the image, identify objects in the thresholded image,
    and generate region props for the image

    Parameters
    ----------
    curr_im : current image being analyzed
    bg_im : background image generated from dataset

    Returns
    -------
    attributes : tuple. The regionproperty fields for many properites of a blob
    labels : tuple. The corresponding label for teh values in attributes
    """

    #If the frame is empty, return nans for frame_attributes
    if np.max(curr_im == 0):
        frame_attributes = np.empty(29)*np.nan
    else:
        frame_nobg = bg_subtract(curr_im, bg_im)

        #find the min-max differential of the bg subtracted image
        #Because both images are normalized between 0 and 1 before
        #subtraction, the background will have a value of zero and
        #any ants should have a pixel intensity of around 1.
        #I will use a min-max differential of 0.5 to identify
        #frames without an ant
        min_max_differential = np.max(frame_nobg)-np.min(frame_nobg)

        #If there is no ant, i.e. min_max_differential <= 0.5
        #return an array of zeros for frame_attributes
        if min_max_differential <= 0.5:
            frame_attributes = np.zeros(29)
        else:
            #normalize bg subtracted image between 0-1
            frame_nobg = normalize_convert_im(frame_nobg)

            frame_thresh,frame_label,label_nums = segment(frame_nobg)

            frame_rp = skimage.measure.regionprops(frame_label,frame_nobg)

            frame_attributes,_ = region_props_to_tuple(frame_rp[0])

    return frame_attributes

#Attempting to write up a function for running the image analyses in parallel
def image_analysis_parallel(bgim_par, im_time_tuple):
    """This function takes as input a background image and a tuple
    containing an image and corresponding timestamp. Calling this function
    using the pool.map function, the iterable, ie the list of tuples containing
    the image and timestampt, will be automatically iterated through, with each
    subsequent entry being passed to this function. The output of this function
    is a pandas dataframe with the image labels as column headings and the entry
    for a single image as the row

    Parameters
    ----------
    bgim_par : a background image for the behavioral arena being used. Generated
                using the jwseg.construct_bg_img function written by Julian
                Wagner

    im_time_tuple : a tuple containing an image array corresponding to a single
                    frame from a behavioral video and the corresponding timestamp

    Returns
    -------
    dataframe :A pandas dataframe with a header will be returned"""

    im_list_par,tmstmp_par = im_time_tuple

    #output_array = ([tmstmp_par, *image_analysis(im_list_par, bgim_par)])

    #return pd.DataFrame(*output_array, columns=['timestamp, s',*image_labels])

    #return output array that contains a list of the timestamp and each of the
    #frame attributes
    return [tmstmp_par, *image_analysis(im_list_par, bgim_par)]


if __name__ == "__main__":
    print('test')
    warnings.filterwarnings('ignore')

    #File path the the video file
    fname = '/data/2019_04_09_LO_1M_sulcatone_run_03\
_cam_0_date_2019_04_09_time_14_49_44_v001.fmf'

    filename = '2019_04_09_LO_1M_sulcatone_run_03'

    #create FMF object of the movie and get image dimensions
    fmf = FMF.FlyMovie(fname)
    frame_width = fmf.get_width()
    frame_height = fmf.get_height()

    image_labels = ['timestamp,s',
                'area',
                'bbox_min_row',
                'bbox_min_col',
                'bbox_max_row',
                'bbox_max_col',
                'bbox_area',
                'centroid_row',
                'centroid_col',
                'convex_area',
                'eccentricity',
                'equivalent_diameter',
                'euler_number',
                'extent',
                'filled_area',
                'label',
                'local_centroid_row',
                'local_centoid_col',
                'major_axis_length',
                'max_intensity',
                'mean_intensity',
                'min_intensity',
                'minor_axis_length',
                'orientation',
                'perimeter',
                'solidity',
                'weighted_centroid_row',
                'weighted_centroid_col',
                'weighted_local_centoid_row',
                'weighted_local_centroid_col']

    #get the number of frames
    frame_num = fmf.get_n_frames()
    frame_num

    print('Loading background image')
    #bg_img_set = []
    #for frame_number in range(3000,10000):
    #    frame,timestamp = fmf.get_frame(frame_number)
    #    bg_img_set.append(frame)

    #bg_img = jwseg.construct_bg_img(bg_img_set)
    #print('Background image completed')

    bg_img = np.loadtxt(filename + '_BGIM.csv')

    #create list that will contain tuples of all images and timestamps
    image_tmstmp_list = []
    print('Loading each frame of the video into RAM')
    #load every frame and timestamp of the video into a list in RAM
    for frame_number in tqdm.tqdm(range(int(frame_num))):
        frame_tmstmp_hold = fmf.get_frame(frame_number)
        image_tmstmp_list.append(frame_tmstmp_hold)

    test_list = image_tmstmp_list[1000:11000]

    #t = time.time()
    #test_output = [[curr_im_time[1], *image_analysis(curr_im_time[0], bg_img)] \
    #           for i,curr_im_time in enumerate(test_list)]

    #test_out_df = pd.DataFrame(test_output,columns = image_labels)
    #elapsed = time.time() - t
    #print('Time to process 10000 images normally: ', elapsed)

    t = time.time()

    #Creating a partial function from image_analysis_parallel
    #The pool.map process can only take a single iterable as input
    #and doesn't take any other arguments. Image_analysis_parallel
    #also requires the background image, which isn't an iterable
    #by creating a partial function, I can feed in the background
    #image to the main function and then feed in the iterable to the
    #partial function
    partial_func = partial(image_analysis_parallel, bg_img)

    #I'm using map instead of imap. Map generates a list that is returned
    #after the iterable has been run through in its entirety. imap
    #saves out to a second iterable over the course of running through the
    #first iterable and is thus available throughout the run.
    with multiprocessing.Pool() as pool:

        #output_data = pool.map(partial_func, image_tmstmp_list)
        output_data = pool.map(partial_func, image_tmstmp_list)

    out_df = pd.DataFrame(output_data,columns = image_labels)
    elapsed = time.time() - t
    print('Time to process all images with multiprocessing: ', elapsed)

    out_df.to_csv('~/sceptovideo/' + filename + '.csv')
