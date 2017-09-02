GREEN = (0,255,0)
BLUE = (255,0,0)
RED = (0,0,255)
ORANGE = (0,128,255)
YELLOW = (0,255,255)
LIGHT_BLUE = (255,255,0)
PURPLE = (205,0,205)
WHITE = (255,255,255)
BLACK = (0,0,0)
switchColor = [(255,255,0),(255,0,255),(0,255,255),(255,0,0),(0,255,0),(0,0,255),(255,128,0),(255,0,128),(128,0,255),(128,255,0),(0,128,255),(0,255,128),(128,128,0),(128,0,128),(0,128,128),(255,64,0),(255,0,64),(64,255,0),(64,0,255),(0,255,64),(0,64,255)]
#switchColor = [(255,0,255),(0,255,255),(255,0,0),(0,255,0),(255,128,0),(255,0,128),(128,0,255),(128,255,0),(0,128,255),(0,255,128),(255,255,0)]


resize_height = 736.0
split_n_row = 1
split_n_column = 1
gaussian_para = 3
small_filter_threshold = (500*30) / float(736*736)

_sharpen = True
_check_overlap = False
_remove_small_and_big = True
_remove_high_density = True
_remove_too_many_edge = True
_checkConvex = False
_gaussian_filter = True
_use_structure_edge = True
_enhance_edge = True
_gray_value_redistribution_local = True
_record_by_csv = False
_use_comebine_weight = False
_combine_two_edge_result_before_filter_obvious = True
_evaluate = True


input_path = '../../input_image/論文實驗影像/一般化偵測/'
#input_path = '../../input_image/論文實驗影像/豆子/'
edge_input_path = '../../edge_input/'
output_path = '../../output_6_8[combine_result_before_filter_obvious]/'
csv_output = '../../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../../evaluate_data/groundtruth_csv/一般化csv/' 

_edge_by_channel = ['bgr_gray']

_showImg = { 'original_image':True, 'original_edge':False, 'enhanced_edge':False, 'original_contour':False, 'contour_filtered':False, 'size':False, 'shape':False, 'color':False, 'cluster_histogram':False , 'original_result':False, 'each_obvious_result':False, 'combine_obvious_result':False, 'obvious_histogram':False, 'each_group_result':False, 'result_obvious':True, 'final_each_group_result':True, 'final_result':False }
_writeImg = { 'original_image':False, 'original_edge':False, 'enhanced_edge':False, 'original_contour':False, 'contour_filtered':False, 'size':False, 'shape':False, 'color':False, 'cluster_histogram':False, 'original_result':False, 'each_obvious_result':False, 'combine_obvious_result':False, 'obvious_histogram':False, 'each_group_result':False, 'result_obvious':False, 'final_each_group_result':False, 'final_result':False }

_show_resize = [ ( 720, 'height' ), ( 1200, 'width' ) ][0]

test_one_img = { 'test':True , 'filename': 'IMG_ (61).jpg' }
#test_one_img = { 'test':True , 'filename': '14_84.png' }

def Parameters():
    
    parameters = {
        'GREEN' : GREEN,
        'BLUE' : BLUE,
        'RED' : RED,
        'ORANGE' : ORANGE,
        'YELLOW' : YELLOW,
        'LIGHT_BLUE' : LIGHT_BLUE,
        'PURPLE' : PURPLE,
        'WHITE' : WHITE,
        'BLACK' : BLACK,
        'switchColor' : switchColor,
        
        'resize_height' : resize_height,
        'split_n_row' : split_n_row,
        'split_n_column' : split_n_column,
        'gaussian_para' : gaussian_para,
        'small_filter_threshold' : small_filter_threshold,
        
        '_sharpen' : _sharpen,
        '_check_overlap' : _check_overlap,
        '_remove_small_and_big' : _remove_small_and_big,
        '_remove_high_density' : _remove_high_density,
        '_remove_too_many_edge' : _remove_too_many_edge,
        '_checkConvex' : _checkConvex,
        '_gaussian_filter' : _gaussian_filter,
        '_use_structure_edge' : _use_structure_edge,
        '_enhance_edge' : _enhance_edge,
        '_gray_value_redistribution_local' : _gray_value_redistribution_local,
        '_record_by_csv' : _record_by_csv,
        '_use_comebine_weight' : _use_comebine_weight,
        '_combine_two_edge_result_before_filter_obvious' : _combine_two_edge_result_before_filter_obvious,
        '_evaluate' : _evaluate, 
        'input_path' : input_path,
        'edge_input_path' : edge_input_path,
        'output_path' : output_path,
        'csv_output' : csv_output,
        'evaluate_csv_path' : evaluate_csv_path,
        
        '_edge_by_channel' : _edge_by_channel,
        
        '_showImg' : _showImg,
        '_writeImg' : _writeImg,
        
        '_show_resize' : _show_resize,
        
        'test_one_img' : test_one_img,   }
    
    
    return parameters
    