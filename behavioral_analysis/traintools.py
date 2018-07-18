import numpy as np

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.themes import Theme
from bokeh.io import show
from bokeh.plotting import figure, output_file, Column
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource
from bokeh.events import DoubleTap
from bokeh.models.widgets import TextInput
from bokeh.palettes import viridis

import segmentation
import bootcamp_utils

def _check_ims(ims):
    if not segmentation._check_array_like(ims):
        raise RuntimeError("The given ims object is not array like, it is " + str(type(ims)))
    if not len(np.array(ims).shape) == 3:
        raise RuntimeError("Need to provide an array with shape (n, m, p). Provided array has shape " +
                           str(np.array(ims).shape))
    [segmentation._check_image_input(im) for im in ims]
    
def point_label(ims):
        
    _check_ims(ims)
    ims = np.array(ims)

    point_labels = ColumnDataSource({'x': [],
                                     'y': [],
                                     'frame': []})
    
    alphas = []

    pal = viridis(len(ims))

    def modify_doc(doc):
        im_num = [0,]

        images = ims.copy()
        plot, source = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]], return_im=True)
        source = source.data_source

        renderer = plot.scatter(x='x', y='y', source=point_labels, color='white', size=3)
        columns = [TableColumn(field="x", title="x"),
                   TableColumn(field="y", title="y"),
                   TableColumn(field='frame', title='frame')]
        table = DataTable(source=point_labels, columns=columns, editable=True, height=200)
        draw_tool = PointDrawTool(renderers=[renderer], empty_value=im_num[-1])
        plot.add_tools(draw_tool)
        plot.toolbar.active_tap = draw_tool

        def callback(attr, old, new):
            im_num.append(int(new))
            _, data = bootcamp_utils.viz.bokeh_imshow(images[int(new)], return_im=True)
            data = data.data_source
            source.data = data.data
            draw_tool.empty_value = im_num[-1]

        def callback_forward():
            if (len(images) - 2) < im_num[-1]:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+1], return_im=True)
            im_num.append(im_num[-1]+1)
            data = data.data_source
            source.data = data.data
            draw_tool.empty_value = im_num[-1]

        def callback_backward():
            if im_num[-1] == 0:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]-1], return_im=True)
            im_num.append(im_num[-1]-1)
            data = data.data_source
            source.data = data.data
            draw_tool.empty_value = im_num[-1]

        slider = Slider(start=0, end=29, value=0, step=1, title="Frame Number")
        button_back = Button(label='back',button_type="success")
        button_forward = Button(label='forward',button_type="success")
        #text_input = TextInput(value="0", title="Frame Num:")
        #text_input.on_change('value', callback)
        slider.on_change('value', callback)
        button_forward.on_click(callback_forward)
        button_back.on_click(callback_backward)

        doc.add_root(column(row(slider), plot, row(button_back, button_forward), table))
    
    show(modify_doc)
    return point_labels

def binary_label(ims, button_left_label='beetle', button_right_label='ant'):
    _check_ims(ims)
    ims = np.array(ims)
    
    frame_labels = ColumnDataSource({'type': [],
                                     'frame': []
                                    })

    def modify_doc(doc):
        im_num = [0,]

        images = ims.copy()
        plot, source = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]], return_im=True)
        source = source.data_source

        columns = [TableColumn(field='type', title='type'),
                   TableColumn(field='frame', title='frame')]
        table = DataTable(source=frame_labels, columns=columns, editable=True, height=200)

        def callback(attr, old, new):
            im_num.append(int(new))
            _, data = bootcamp_utils.viz.bokeh_imshow(images[int(new)], return_im=True)
            data = data.data_source
            source.data = data.data

        def callback_forward():
            if (len(images) - 2) < im_num[-1]:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+1], return_im=True)
            im_num.append(im_num[-1]+1)
            data = data.data_source
            source.data = data.data

        def callback_backward():
            if im_num[-1] == 0:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]-1], return_im=True)
            im_num.append(im_num[-1]-1)
            data = data.data_source
            source.data = data.data

        def callback_label_L():        
            new_data = {'type': [button_left_label],
                        'frame': [im_num[-1]]}
            frame_labels.stream(new_data)

            if (len(images) - 2) < im_num[-1]:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+1], return_im=True)
            im_num.append(im_num[-1]+1)
            data = data.data_source
            source.data = data.data

        def callback_label_R():
            new_data = {'type': [button_right_label],
                        'frame': [im_num[-1]]}
            frame_labels.stream(new_data)

            if (len(images) - 2) < im_num[-1]:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+1], return_im=True)
            im_num.append(im_num[-1]+1)
            data = data.data_source
            source.data = data.data


        slider = Slider(start=0, end=29, value=0, step=1, title="Frame Number")
        button_back = Button(label='back',button_type="success")
        button_forward = Button(label='forward',button_type="success")
        slider.on_change('value', callback)
        button_forward.on_click(callback_forward)
        button_back.on_click(callback_backward)

        button_ID_left = Button(label=button_left_label, button_type='success')
        button_ID_left.on_click(callback_label_L)

        button_ID_right = Button(label=button_right_label, button_type='success')
        button_ID_right.on_click(callback_label_R)



        doc.add_root(column(row(slider), plot,
                            row(button_back, button_forward),
                            row(button_ID_left, button_ID_right),
                            table))

    show(modify_doc)
    return frame_labels