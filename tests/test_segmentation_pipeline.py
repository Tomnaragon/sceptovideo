import numpy as np
import sys
sys.path.append("../behavioral_analysis")
import segmentation
import pytest
import skimage.filters

# test functions for simple segmentation based tracking code

def test_im_shape():
    im = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_im_data_type_list():
    im = [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'list'>")
    
def test_im_data_type_string():
    im = '[[1, 2, 3], [1, 2, 3]]'
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'str'>")
    
def test_im_shape_segment():
    im = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_im_data_type_list_segment():
    im = [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'list'>")
    
def test_im_data_type_string_segment():
    im = '[[1, 2, 3], [1, 2, 3]]'
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'str'>")
    
def test_provided_function_callable():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im, thresh_func='Hello, world.')
    excinfo.match("The provided function is not callable")
    
def test_provided_function_callable():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    args = (3)
    assert segmentation._check_function_input(im, thresh_func=skimage.filters.threshold_local, *args) == True

def test_provided_function_returns_float():
    
    pass