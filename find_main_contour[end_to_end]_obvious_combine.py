import numpy as np
import cv2
import os 
import time
import csv
import math
import get_contour_feature
import operator
from operator import itemgetter
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt




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
#check if combine the ouput before the obviousity filter
_combine_two_edge_result_before_filter_obvious = True
_evaluate = False


input_path = './input/image/'
# structure forest output
edge_input_path = './input/edge_image/'
output_path = './output/'

csv_output = '../../output_csv_6_8[combine_result_before_filter_obvious]/'
evaluate_csv_path = '../../evaluate_data/groundtruth_csv/¤@¯ë¤Æcsv/' 

_edge_by_channel = ['bgr_gray']

_showImg = { 'original_image':True, 'original_edge':False, 'enhanced_edge':False, 'original_contour':True, 'contour_filtered':True, 'size':True, 'shape':True, 'color':True, 'cluster_histogram':False , 'original_result':True, 'each_obvious_result':True, 'combine_obvious_result':True, 'obvious_histogram':False, 'each_group_result':True, 'result_obvious':True, 'final_each_group_result':True, 'final_result':False }
_writeImg = { 'original_image':False, 'original_edge':False, 'enhanced_edge':False, 'original_contour':False, 'contour_filtered':False, 'size':False, 'shape':False, 'color':False, 'cluster_histogram':False, 'original_result':False, 'each_obvious_result':False, 'combine_obvious_result':False, 'obvious_histogram':False, 'each_group_result':False, 'result_obvious':False, 'final_each_group_result':False, 'final_result':False }

_show_resize = [ ( 720, 'height' ), ( 1200, 'width' ) ][0]

test_one_img = { 'test':True , 'filename': 'IMG_ (34).jpg' }
#test_one_img = { 'test':True , 'filename': '14_84.png' }

def main():
    
    
   
    switch_i = 0

    max_time_img = ''
    min_time_img = ''
    min_time = 99999.0
    max_time = 0.0
    
    evaluation_csv = [['Image name','TP','FP','FN','Precision','Recall','F_measure','Error_rate']]
    
    #line88 - line113 input exception
    for i,fileName in enumerate(os.listdir(input_path)):
        
        
        if(fileName[-3:]!='jpg' and fileName[-3:]!='JPG' and fileName[-4:]!='jpeg' and fileName[-3:]!='png'):
            print "Wrong format file: "+fileName
            continue   
        
        start_time = time.time()
        
        if test_one_img['test'] and i > 0 :
            break
        
        if test_one_img['test']:
            fileName =  test_one_img['filename'] 
        
        print 'Input:',fileName
        
        if not os.path.isfile( input_path + fileName ):
            print input_path + fileName   
            print 'FILE does not exist!'
            break
        
        #===========================
        #decide whcih method (canny/structure forest) 
        _use_structure_edge = True
        #============================    
        
        if not os.path.isfile( edge_input_path + fileName[:-4] + '_edge.jpg' ) and _use_structure_edge:
            print edge_input_path + fileName[:-4] + '_edge.jpg'
            print 'EDGE FILE does not exist!'
            break
        
        # read color image
        color_image_ori = cv2.imread( input_path + fileName )    
    
        height, width = color_image_ori.shape[:2]
        image_resi = cv2.resize( color_image_ori, (0,0), fx= resize_height/height, fy= resize_height/height)        
      
        if _showImg['original_image']:
            cv2.imshow( fileName + ' origianl_image', ShowResize(image_resi) )
            cv2.waitKey(100)
        if _writeImg['original_image']:
            cv2.imwrite(output_path+fileName[:-4]+'_a_original_image.jpg', image_resi )        
           
        
        final_differ_edge_group = []
        
        #check if two edge detection method is both complete
        twice = False
        for j in xrange(2):
            
            edge_type = 'structure'
                     
            if _use_structure_edge :
                            
                # read edge image from matlab 
                edge_image_ori = cv2.imread( edge_input_path + fileName[:-4] + '_edge.jpg' , cv2.IMREAD_GRAYSCALE ) 
                height, width = edge_image_ori.shape[:2]
                edged = cv2.resize( edge_image_ori, (0,0), fx= resize_height/height, fy= resize_height/height)                          
                #thresh_gray,edged = cv2.threshold(edge_image_resi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
                
            else:
                edge_type = 'canny'
              
                #filter the noise 
                if _gaussian_filter :  
                    print 'Gaussian filter'
                    image_resi = cv2.GaussianBlur(image_resi, (gaussian_para, gaussian_para),0)
                
                if _sharpen :
                    print 'Sharpening'
                    image_resi = Sharpen(image_resi)
            
                re_height, re_width = image_resi.shape[:2]
            
                offset_r = re_height/split_n_row
                offset_c = re_width/split_n_column
                
                print 'Canny Detect edge'
                edged = np.zeros(image_resi.shape[:2], np.uint8) 
                
                for row_n in np.arange(0,split_n_row,0.5):
                    for column_n in np.arange(0,split_n_column,0.5):
                        
                        r_l =  int(row_n*offset_r)
                        r_r = int((row_n+1)*offset_r)
                        c_l = int(column_n*offset_c)
                        c_r = int((column_n+1)*offset_c)
                        
                        if row_n == split_n_row-0.5 :
                            r_r = int(re_height)
                        if column_n == split_n_column-0.5 :
                            c_r = int(re_width)    
                                                   
                        BGR_dic, HSV_dic, LAB_dic = SplitColorChannel( image_resi[ r_l : r_r , c_l : c_r ] )
                                               
                        channel_img_dic = { 'bgr_gray':BGR_dic['img_bgr_gray'], 'b':BGR_dic['img_b'], 'g':BGR_dic['img_g'], 'r':BGR_dic['img_r'], 'h':HSV_dic['img_h'], 's':HSV_dic['img_s'], 'v':HSV_dic['img_v'], 'l':LAB_dic['img_l'], 'a':LAB_dic['img_a'], 'b':LAB_dic['img_b'] }
                        channel_thre_dic = { 'bgr_gray':BGR_dic['thre_bgr_gray'], 'b':BGR_dic['thre_b'], 'g':BGR_dic['thre_g'], 'r':BGR_dic['thre_r'], 'h':HSV_dic['thre_h'], 's':HSV_dic['thre_s'], 'v':HSV_dic['thre_v'], 'l':LAB_dic['thre_l'], 'a':LAB_dic['thre_a'], 'b':LAB_dic['thre_b'] }
                        
                        for chan in _edge_by_channel:
                            if channel_thre_dic[chan] > 20 :
                                edged[ r_l : r_r , c_l : c_r ] = edged[ r_l : r_r , c_l : c_r ] | cv2.Canny( channel_img_dic[chan], 0.5*channel_thre_dic[chan], channel_thre_dic[chan] )
                #image_resi = cv2.resize( image_resi, (0,0), fx= 1.0/scale, fy= 1.0/scale)
                                            
            # end detect edge else  
    
            if _showImg['original_edge']:
                cv2.imshow( fileName + ' origianl_edge['+str(edge_type)+']', ShowResize(edged) )
                cv2.waitKey(100)
            if _writeImg['original_edge']:
                cv2.imwrite(output_path+fileName[:-4]+'_b_original_edge['+str(edge_type)+'].jpg', edged )                 
            
            if _enhance_edge and _use_structure_edge:
                # enhance and close the edge
                print 'Enhance edge'
                #local equalization
                #refer to : http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
                if _gray_value_redistribution_local : 
                    
                    # create a CLAHE object (Arguments are optional).
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    edged = clahe.apply(edged)                    
                 
                else:
                    #golbal equalization
                    print 'Equalization'
                    edged = cv2.equalizeHist(edged)            
                
                if _showImg['enhanced_edge']:
                    cv2.imshow( fileName + ' enhanced_edge['+str(edge_type)+']', ShowResize(edged) )
                    cv2.waitKey(100)
                if _writeImg['enhanced_edge']:
                    cv2.imwrite( output_path + fileName[:-4] +'_c_enhanced_edge['+str(edge_type)+'].jpg', edged )                 
            # end enhance edge if
            
            if _use_structure_edge :
                _use_structure_edge = False
            else:
                _use_structure_edge = True                             
                           
                           
            print 'Find countour'     
            edged = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            #==============
            # contour detection 
            contours = cv2.findContours(edged,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]
            #==============
            contour_image = np.zeros( image_resi.shape, np.uint8 )
          
            color_index = 0 
            
            for c in contours :
                COLOR = switchColor[ color_index % len(switchColor) ]     
                color_index += 1
                cv2.drawContours( contour_image, [c], -1, COLOR, 2 )            
            
            if _showImg['original_contour']:
                cv2.imshow( fileName + ' original_contour['+str(edge_type)+']', ShowResize(contour_image) )
                cv2.waitKey(100) 
            if _writeImg['original_contour']:
                cv2.imwrite( output_path + fileName[:-4] +'_d_original_contour['+str(edge_type)+'].jpg', contour_image )                 
             
                 
            tmp_cnt_list = [contours[0]]
            tmp_cnt = contours[0]
            # check isOverlap
            # Since if is overlap , the order of the overlapped contours will be continuous 
            for c in contours[1:]:
                if not IsOverlap(tmp_cnt,c):
                    tmp_cnt_list.append(c)
                tmp_cnt = c
                
            contours = tmp_cnt_list
            
            noise = 0  
            contour_list = []
            re_height, re_width = image_resi.shape[:2]
            
            # line264 - line 376 Find Contour and Filter
            print 'Filter contour'
            print '------------------------'
            
            # decide if use small contour filter
            small_cnt_cover_area = 0.0
            small_cnt_count = 0
            for c in contours:
                cnt_area = max( len(c), cv2.contourArea(c) )
                if cnt_area < 60 :
                    small_cnt_cover_area += cnt_area
                    small_cnt_count += 1
                
            #print 'cover rate:',small_cnt_cover_area / float(re_height*re_width),small_cnt_cover_area , float(re_height*re_width)
            #print 'threshold:',small_filter_threshold
            #if small_cnt_cover_area / float(re_height*re_width) > small_filter_threshold :
            
            # normal pic for small noise more than 500
            if small_cnt_count > 500:
                cnt_min_size = 60
            # colony pic for small noise less than 500 (400 for colonies and 100 for error tolerance)
            else:
                cnt_min_size = 4
            
            # remove contours by some limitations    
            for c in contours:
            
                #CountCntArea(c,image_resi)
                
                if _remove_small_and_big :
                    # remove too small or too big contour
                    # contour perimeter less than 1/3 image perimeter 
                    if len(c) < cnt_min_size or len(c) > (re_height+re_width)*2/3.0: 
                        continue        
                
                if _checkConvex :
                    # remove contour which is not Convex hull
                    #print 'Check convexhull'
                    if not cv2.isContourConvex(np.array(c)):
                        contour_image = np.zeros( image_resi.shape, np.uint8 )
                        cv2.drawContours( contour_image, [c], -1, GREEN, 1 ) 
                        cv2.imshow( fileName + ' countour', ShowResize(contour_image) )
                        cv2.waitKey(100)                          
                        continue
                
                if _remove_high_density :
                    # remove contour whose density is too large or like a line
                    #print 'Remove contour with high density'
                    area = cv2.contourArea(c) 
                    shape_factor = 4*np.pi*area / float( pow(len(c), 2 ) )
                    if cv2.contourArea(cv2.convexHull(c)) == 0:
                        continue
                    solidity =  area / cv2.contourArea(cv2.convexHull(c)) 
                    if area < 4 or solidity < 0.5 :
                    #if area < 4 or ( len(c) > 30 and float(len(c)) / area > 0.5 ) or ( len(c) <= 30 and float(len(c)) / area > 0.75 ): 
                    #if area < 4 or shape_factor < 0.1 :
                        noise+=1
                        continue
                
                if _remove_too_many_edge :
                    # remove contour which has too many edge
                    #print 'Remove contour with too many edges '
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 10, True)
                    #contour_image = np.zeros( image_resi.shape, np.uint8 )
                    #cv2.drawContours( contour_image, [c], -1, GREEN, 1 )
                    #cv2.imshow('edge number : '+str(len(approx)), ShowResize(contour_image))
                    #cv2.waitKey(100)
                    if len(approx) > 50 : 
                        continue
   
                contour_list.append(c)
            # end filter contour for
            
            contour_dic_list = []
            for cnt in contour_list:
                contour_dic_list.append( {'cnt':cnt} )
            # remove outer contour of two overlapping contours whose sizes are close          
            if _check_overlap :     
                print 'Remove overlap contour keep inner ones'
                contour_dic_list = CheckOverlap(contour_dic_list)
            
            contour_list = []
            for cnt_dic in contour_dic_list:
                contour_list.append(cnt_dic['cnt'])
                
            if len(contour_list) == 0 :
                continue
            
            print '------------------------'
                       
            #
            # draw contour by different color
            contour_image = np.zeros( image_resi.shape, np.uint8 )
            contour_image[:] = BLACK
            color_index = 0 
            for c in contour_list :
                COLOR = switchColor[ color_index % len(switchColor) ]     
                color_index += 1
                cv2.drawContours( contour_image, [c], -1, COLOR, 2 )
                #cv2.drawContours( gradient_img_copy, [c], -1, COLOR, 1 )
            
            #combine_image = np.concatenate( (gradient_img, gradient_img_copy), axis = 1)
            
            #cv2.imwrite( './gradient.jpg',combine_image)            
            #cv2.imshow( 'gradient',combine_image)
            #cv2.waitKey(100)
    
            if _showImg['contour_filtered']:
                cv2.imshow( fileName + ' contour_filtered['+str(edge_type)+']', ShowResize(contour_image) )
                cv2.waitKey(100)
            if _writeImg['contour_filtered']:
                cv2.imwrite( output_path + fileName[:-4] +'_e_contour_filtered['+str(edge_type)+'].jpg', contour_image )

            print 'Extract contour feature'
            # line 382 - line 520 feature extraction and cluster
            # Get contour feature
            c_list, cnt_shape_list, cnt_color_list, cnt_size_list, cnt_color_gradient_list = get_contour_feature.extract_feature( image_resi, contour_list )

            cnt_dic_list = []
            for i in range( len(c_list) ):
                cnt_dic_list.append( { 'cnt':c_list[i], 'shape':cnt_shape_list[i], 'color':cnt_color_list[i], 'size':cnt_size_list[i], 'color_gradient':cnt_color_gradient_list[i] } )
            
            feature_dic = { 'cnt':c_list, 'shape':cnt_shape_list, 'color':cnt_color_list, 'size':cnt_size_list }
        
            para = [ 'size', 'shape' , 'color' ] 
            
            # total contour number
            cnt_N = len(c_list)
            
            if cnt_N < 1:
                print 'No any contour!'
                continue            
            
            label_list_dic = {}
            
            print 'Respectively use shape, color, and size as feature set to cluster'
            # Respectively use shape, color, and size as feature set to cluster
            for para_index in xrange( len(para) ):
                
                print 'para:',para[para_index]
   
                contour_feature_list =  feature_dic[para[para_index]]
                
                # hierarchical clustering
                # output the classified consequence
                label_list = Hierarchical_clustering( contour_feature_list, fileName, para[para_index], edge_type )   
                
                unique_label, label_counts = np.unique(label_list, return_counts=True)
                
                # draw contours of each group refer to the result clustered by size, shape or color
                contour_image = np.zeros(image_resi.shape, np.uint8)
                contour_image[:] = BLACK                      
                color_index = 0    
                for label in unique_label :
                    COLOR = switchColor[ color_index % len(switchColor) ]
                    color_index += 1
                    tmp_splited_group = []
                    for i in xrange( len(label_list) ):
                        if label_list[i] == label :
                            tmp_splited_group.append( c_list[i] )                        
                    cv2.drawContours( contour_image, np.array(tmp_splited_group), -1, COLOR, 2 )
                
                if _showImg[para[para_index]]:
                    cv2.imshow( 'cluster by :'+ para[para_index]+'['+str(edge_type)+']', ShowResize(contour_image) )
                    cv2.waitKey(100)
                if _writeImg[para[para_index]]:
                    cv2.imwrite( output_path + fileName[:-4] +'_f_para['+para[para_index]+']_['+str(edge_type)+'].jpg', contour_image ) 
                
                # save the 3 types of the classified output
                label_list_dic[para[para_index]] = label_list
                
            # end para_index for
            
            # intersect the label clustered by size, shpae, and color
            # ex: [0_1_1 , 2_0_1]
            combine_label_list = []
            for i in xrange( cnt_N ):
                combine_label_list.append( str(label_list_dic['size'][i]) + '_' + str(label_list_dic['shape'][i]) + '_' + str(label_list_dic['color'][i])  )
                
            unique_label, label_counts = np.unique(combine_label_list, return_counts=True)      
            label_dic = dict(zip(unique_label, label_counts))
            max_label = max( label_dic.iteritems(), key=operator.itemgetter(1) )[0]
       
            # find the final group by the intersected label and draw
            final_group = []  
            contour_image = np.zeros(image_resi.shape, np.uint8)
            contour_image[:] = BLACK     
            contour_image_max = np.zeros(image_resi.shape, np.uint8)
            contour_image_max[:] = BLACK 
            
           
            color_index = 0             
            for label in unique_label :
                contour_image_each = image_resi.copy()
                # darken the image to make the contour visible
                contour_image_each[:] = contour_image_each[:]/3.0
                COLOR = switchColor[ color_index % len(switchColor) ]
                color_index += 1
                tmp_group = []
                for i in xrange( cnt_N ):
                    if combine_label_list[i] == label :
                        tmp_group.append( cnt_dic_list[i] ) 
                
                #tmp_group = CheckOverlap(tmp_group)
                tmp_cnt_group = []
                avg_color_gradient = 0.0
                avg_shape_factor = 0.0
                tmp_area = 0.0
                
                
                #for each final group count obvious factor
                for cnt_dic in tmp_group:
                    cnt = cnt_dic['cnt']
                    cnt_area = cv2.contourArea(cnt)
                    tmp_area += cnt_area
                    avg_shape_factor += (4*np.pi*cnt_area)/float(pow(len(cnt),2))
                    #print cnt_dic['color_gradient'],Eucl_distance(img_lab,cnt_dic['color']),max( cnt_dic['color_gradient'], Eucl_distance(img_lab,cnt_dic['color']))
                    #avg_color_gradient += max( cnt_dic['color_gradient'], Eucl_distance(img_lab,cnt_dic['color']))
                    avg_color_gradient += cnt_dic['color_gradient']
                    #avg_color_gradient += Eucl_distance(img_lab,cnt_dic['color'])
                    tmp_cnt_group.append(cnt)
                
                avg_shape_factor /= float(len(tmp_group))
                avg_color_gradient /= float(len(tmp_group))
                
                if len(tmp_cnt_group) < 2 :
                    continue
                
                #if label == max_label :                  
                    #cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, RED, 2 )
                #else:
                    #cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, GREEN, 1 ) 
            
                cv2.drawContours( contour_image, np.array(tmp_cnt_group), -1, COLOR, 2 )
                cv2.drawContours( contour_image_each, np.array(tmp_cnt_group), -1, COLOR, 2 )
                
                #print 'color_gradient:',avg_color_gradient
                final_group.append( { 'cnt':tmp_cnt_group, 'cover_area':tmp_area, 'color_gradient':avg_color_gradient, 'shape_factor':avg_shape_factor, 'obvious_weight':0, 'combine_weight':0.0, 'group_dic':tmp_group } )
                
                contour_image_each = cv2.resize( contour_image_each, (0,0), fx = float(color_image_ori.shape[0])/contour_image_each.shape[0], fy = float(color_image_ori.shape[0])/contour_image_each.shape[0])
                #print 'shape_factor:',avg_shape_factor
                #cv2.imshow(fileName+' shape_factor['+str(avg_shape_factor)+']', ShowResize(contour_image_each) )
                #cv2.waitKey(100)                
            # end find final group for
            # sort the group from the max area to min group and get max count
           
            if _showImg['original_result']:
                cv2.imshow(fileName+' original_result['+str(edge_type)+']', ShowResize(contour_image) )
                cv2.waitKey(100)     
            if _writeImg['original_result']:
                cv2.imwrite( output_path + fileName[:-4] +'_g_original_result['+str(edge_type)+'].jpg', contour_image )            
                
            if len(final_group) < 1:
                print 'No any pattern'
                continue
        
            #====================================================================================
            
            #img_lab = [0.0,0.0,0.0]
            #lab = cv2.cvtColor( image_resi, cv2.COLOR_BGR2LAB)
            #for tmp_h in range( re_height ): 
                #for tmp_w in range( re_width ):
                    #img_lab[0] += lab[tmp_h,tmp_w][0]
                    #img_lab[1] += lab[tmp_h,tmp_w][1]
                    #img_lab[2] += lab[tmp_h,tmp_w][2]      
            #img_lab[0] /= float(re_height*re_width)
            #img_lab[1] /= float(re_height*re_width)
            #img_lab[2] /= float(re_height*re_width)                    
            
            # line 536 - line 632 combine two edge detection results
            if _combine_two_edge_result_before_filter_obvious :
                for f_edge_group in final_group:
                    final_differ_edge_group.append(f_edge_group)
                                      
                if not twice :
                    twice = True
                    continue
                
                # check two edge contour overlap
                compare_overlap_queue = []
                total_group_number = len( final_differ_edge_group )
                
                for group_index in range( total_group_number ):
                    cnt_group = final_differ_edge_group[group_index]['group_dic']
            
                    for cnt_dic in cnt_group:
                        compare_overlap_queue.append( { 'cnt':cnt_dic['cnt'], 'label':group_index, 'group_weight':len(cnt_group), 'cnt_dic':cnt_dic  } )
                
                _label = [x['label'] for x in compare_overlap_queue]
                print 'label_dic:',[(y,_label.count(y)) for y in set(_label)]
                
                compare_overlap_queue = CheckOverlap( compare_overlap_queue, keep = 'group_weight' )  
                
                contour_image[:] = BLACK
                _label = [x['label'] for x in compare_overlap_queue]
                
                print 'label_dic:',[(y,_label.count(y)) for y in set(_label)]                
                final_group = []            
                
                for label_i in range( total_group_number ):
                #for label in unique_label :
                    contour_image_each = image_resi.copy()
                    # darken the image to make the contour visible
                    contour_image_each[:] = contour_image_each[:]/3.0
                    COLOR = switchColor[ color_index % len(switchColor) ]
                    color_index += 1
                    tmp_group = []
                    for i in xrange( len(compare_overlap_queue) ):
                        #print 'compare_overlap_queue , label_i : ', compare_overlap_queue[i]['label'] , label_i
                        if compare_overlap_queue[i]['label'] == label_i :
                            tmp_group.append( compare_overlap_queue[i]['cnt_dic'] ) 
                            #cv2.drawContours( contour_image, np.array([compare_overlap_queue[i]['cnt']]), -1, GREEN, 2 )
                    
                    if len(tmp_group) < 1 :
                        continue                    

                    tmp_cnt_group = []
                    avg_color_gradient = 0.0
                    avg_shape_factor = 0.0
                    tmp_area = 0.0
                    
                    
                    
                    #for each final group count obvious factor
                    for cnt_dic in tmp_group:
                        cnt = cnt_dic['cnt']
                        cnt_area = cv2.contourArea(cnt)
                        tmp_area += cnt_area
                        #avg_shape_factor += (4*np.pi*cnt_area)/float(pow(len(cnt),2))
                        avg_shape_factor += cnt_area/float(cv2.contourArea(cv2.convexHull(cnt)))
                        #avg_color_gradient += max( cnt_dic['color_gradient'], Eucl_distance(img_lab,cnt_dic['color']))
                        avg_color_gradient += cnt_dic['color_gradient']
                        tmp_cnt_group.append(cnt)
                    
                    avg_shape_factor /= float(len(tmp_group))
                    avg_color_gradient /= float(len(tmp_group))
                    avg_area = tmp_area / float(len(tmp_group))
                    
                    if len(tmp_cnt_group) < 2 :
                        continue
                    
                    
                    
                    #if label == max_label :                  
                        #cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, RED, 2 )
                    #else:
                        #cv2.drawContours( contour_image_max, np.array(tmp_cnt_group), -1, GREEN, 1 ) 
                
                    cv2.drawContours( contour_image, np.array(tmp_cnt_group), -1, COLOR, 2 )
                    cv2.drawContours( contour_image_each, np.array(tmp_cnt_group), -1, COLOR, 2 )
                    
                    
                    final_group.append( { 'cnt':tmp_cnt_group, 'avg_area':avg_area, 'cover_area':tmp_area, 'color_gradient':avg_color_gradient, 'shape_factor':avg_shape_factor, 'obvious_weight':0, 'combine_weight':0.0, 'group_dic':tmp_group } )
                    
                    contour_image_each = cv2.resize( contour_image_each, (0,0), fx = float(color_image_ori.shape[0])/contour_image_each.shape[0], fy = float(color_image_ori.shape[0])/contour_image_each.shape[0])
                    #print 'shape_factor:',avg_shape_factor
                    #cv2.imshow(fileName+' shape_factor['+str(label_i)+']', ShowResize(contour_image_each) )
                    #cv2.waitKey(100)                
                # end find final group for  
                
                if _showImg['original_result']:
                    cv2.imshow(fileName+' remove_overlap_combine_cnt', ShowResize(contour_image) )
                    cv2.waitKey(100)     
                if _writeImg['original_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_g_remove_overlap_combine_cnt.jpg', contour_image )                
             
            # end _combine_two_edge_result_before_filter_obvious if 
            #====================================================================================
            
            #print "@@ len(final_group):",len(final_group)
            
            # line 637 - line 712 obviousity filter
            obvious_list = ['cover_area','color_gradient','shape_factor']
            #sort final cnt group by cover_area , shape_factor and color_gradient
            for obvious_para in obvious_list:
                
                if obvious_para == 'color_gradient':
                    avg_img_gradient = Avg_Img_Gredient(image_resi)
                    final_group.append( { 'cnt':[], 'cover_area':[], 'color_gradient':avg_img_gradient, 'shape_factor':[], 'obvious_weight':-1, 'combine_weight':-1 } )
                    #print 'avg_img_gradient:',avg_img_gradient
                    
                final_group.sort( key = lambda x:x[obvious_para], reverse = True )
                obvious_index = len(final_group)-1
                max_diff = 0
                area_list = [ final_group[0][obvious_para] ]
                
                if final_group[0]['combine_weight'] < 0 :
                    final_group.remove({ 'cnt':[], 'cover_area':[], 'color_gradient':avg_img_gradient, 'shape_factor':[], 'obvious_weight':-1, 'combine_weight':-1 })
                    print 'No color_gradient result'
                    continue
                    
                final_group[0]['combine_weight'] += 1.0
                    
                for i in range( 1, len( final_group ) ):
                    area_list.append(final_group[i][obvious_para])
                    diff = final_group[i-1][obvious_para] - final_group[i][obvious_para] 
                    #diff = float(final_group[i-1][obvious_para]) / sum([x[obvious_para] for x in final_group[i:]])
                    if final_group[i]['combine_weight'] != -1 :
                        final_group[i]['combine_weight'] += final_group[i][obvious_para]/float(final_group[0][obvious_para])
                    
                    if diff > max_diff:
                    #if 0.8*final_group[i-1][obvious_para] > final_group[i][obvious_para] and diff > max_diff:
                        if obvious_para == 'cover_area' and 0.5*final_group[i-1][obvious_para] < final_group[i][obvious_para] :
                            continue
            
                        max_diff = diff
                        obvious_index = i-1
                   
                #print obvious_para,'_list:',area_list
                print 'obvious_index:',obvious_index
                #contour_image[:] = BLACK
                
                for i in range( obvious_index+1 ):
                    if final_group[i]['obvious_weight'] == -1:
                        obvious_index = i
                        break

                    final_group[i]['obvious_weight'] += 1
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, GREEN, 2 )
                    #cv2.imshow('fish:'+str(final_group[i]['obvious_weight']), contour_image)
                    #cv2.waitKey(0)    
                    
                for i in range( obvious_index+1, len(final_group) ):
                    COLOR = RED
                    if  obvious_para == 'shape_factor' and final_group[i]['shape_factor'] >= 0.8 :
                        COLOR = GREEN
                        final_group[i]['obvious_weight'] += 1
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, COLOR, 2 )  
                    
                    
                if _showImg['each_obvious_result']:
                    cv2.imshow(fileName+' obvious_para:['+obvious_para+'] | Green for obvious['+str(edge_type)+']', ShowResize(contour_image) )
                    cv2.waitKey(100)     
                if _writeImg['each_obvious_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_h_para['+obvious_para+']_obvious(Green)['+str(edge_type)+'].jpg', contour_image )   
                
                plt.bar(left=range(len(area_list)),height=area_list)   
                plt.title( obvious_para+' cut_point : '+str(obvious_index)+'  | value: '+str(final_group[obvious_index][obvious_para]) + '  |[' + str(edge_type) + ']' )
                       
                if _showImg['obvious_histogram']:
                    plt.show()
                if _writeImg['obvious_histogram']:
                    plt.savefig(output_path+fileName[:-4]+'_h_para['+obvious_para+']_obvious_his['+str(edge_type)+'].png')    
                plt.close()  
                
                if obvious_para == 'color_gradient':
                    final_group.remove({ 'cnt':[], 'cover_area':[], 'color_gradient':avg_img_gradient, 'shape_factor':[], 'obvious_weight':-1, 'combine_weight':-1 })
                
            # end obvious para for
            
            final_obvious_group = []
            # "take the sum of 3 obviousity attribute" to decide which  groups to remain 
            if _use_comebine_weight :
                final_group.sort( key = lambda x:x['combine_weight'], reverse = True )
                obvious_index = len(final_group)-1
                max_diff = 0
                area_list = [ final_group[0]['combine_weight'] ]
                
                for i in range( 1, len( final_group ) ):
                    area_list.append(final_group[i]['combine_weight'])
                    diff = final_group[i-1]['combine_weight'] - final_group[i]['combine_weight']
                    
                    if final_group[i-1]['combine_weight'] > final_group[i]['combine_weight'] and diff > max_diff:
                        max_diff = diff
                        obvious_index = i-1
                   
                print obvious_para,'_list:',area_list  
                
                for i in range(obvious_index+1):
                    final_obvious_group.append(final_group[i])
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, GREEN, 2 )
                for i in range(obvious_index+1,len(final_group)):
                    cv2.drawContours( contour_image, np.array(final_group[i]['cnt']), -1, RED, 2 )  
                    
               
                if _showImg['combine_obvious_result']:
                    cv2.imshow(fileName+' combine_obvious_result | Green for obvious', ShowResize(contour_image) )
                    cv2.waitKey(100)     
                if _writeImg['combine_obvious_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_combine_obvious_result(Green).jpg', contour_image )   
                
                plt.bar(left=range(len(area_list)),height=area_list)   
                plt.title( 'combine_obvious_result cut_point : '+str(obvious_index)+'  | value: '+str(final_group[obvious_index][obvious_para]) )
                       
                if _showImg['obvious_histogram']:
                    plt.show()
                if _writeImg['obvious_histogram']:
                    plt.savefig(output_path+fileName[:-4]+'_combine_obvious_result_his.png')    
                plt.close()                  
                
            # "vote (0-3) " to decide which groups to remain
            else:
                
                final_group.sort( key = lambda x:x['obvious_weight'], reverse = True )
                #if final_group[0]['obvious_weight'] == 3:
                    #weight = 3
                weight = final_group[0]['obvious_weight']
                #print 'weight:',weight
                
                for f_group in final_group :
                          
                    # determine obvious if match more than two obvious condition 
                    if f_group['obvious_weight'] == weight:
                        final_obvious_group.append(f_group)
                                                                
                        
            # end choose obvious way if 
            
            # search other contour by previous found obvious contours
            #-------------------------------------------------------------------------------------
            
            #final_obvious_group = Search_whole_Img_by_Obvious_Cnt( image_resi, final_obvious_group )
           
            #-------------------------------------------------------------------------------------
            if not _combine_two_edge_result_before_filter_obvious :
                
                contour_image = image_resi.copy()
                contour_image[:] = contour_image[:]/3.0
                for tmp_group in final_obvious_group:
                    tmp_group = tmp_group['cnt']
                    
                    if len(tmp_group) < 2 :
                        continue
                    
                    
                    contour_image_each = image_resi.copy()
                    # darken the image to make the contour visible
                    contour_image_each[:] = contour_image_each[:]/3.0                
                    COLOR = switchColor[ color_index % len(switchColor) ]
                    color_index += 1                
                    cv2.drawContours( contour_image, np.array(tmp_group), -1, COLOR, 2 )
                    cv2.drawContours( contour_image_each, np.array(tmp_group), -1, COLOR, 2 )
        
                    if _showImg['each_group_result']:            
                            cv2.imshow(fileName+' each_group_result_label['+str(color_index)+']_Count['+str(len(tmp_group))+']_['+str(edge_type)+']', ShowResize(contour_image_each) )
                            cv2.waitKey(100)     
                    if _writeImg['each_group_result']:
                            cv2.imwrite( output_path + fileName[:-4] +'_i_label['+str(color_index)+']_Count['+str(len(tmp_group))+']_['+str(edge_type)+'].jpg', contour_image_each )  
        
        
                contour_image = cv2.resize( contour_image, (0,0), fx = height/resize_height, fy = height/resize_height)
                combine_image = np.concatenate((color_image_ori, contour_image), axis=1) 
        
                if _showImg['result_obvious']:
                    cv2.imshow(fileName+' result_obvious['+str(edge_type)+']', ShowResize(combine_image) )
                    cv2.waitKey(100)     
                if _writeImg['result_obvious']:
                    cv2.imwrite( output_path + fileName[:-4] +'_j_result_obvious['+str(edge_type)+'].jpg', combine_image )            
    
    
                for f_group in final_obvious_group :
                    final_differ_edge_group.append(f_group)
                            
            else:
                final_differ_edge_group = final_obvious_group
           
        #end scale for
        
        final_nonoverlap_cnt_group = []
        compare_overlap_queue = []
        total_group_number = len( final_differ_edge_group )
        # get all group cnt and filter overlap 
        for group_index in range( total_group_number ) :
            cnt_group = final_differ_edge_group[group_index]['group_dic']
            
            for cnt_dic in cnt_group:
                compare_overlap_queue.append( { 'cnt':cnt_dic['cnt'], 'label':group_index, 'group_weight':len(cnt_group), 'color':cnt_dic['color']  } )
        
        if not _combine_two_edge_result_before_filter_obvious :
            compare_overlap_queue = CheckOverlap( compare_overlap_queue, keep = 'group_weight' )
       
        for label_i in range( total_group_number ) :
            tmp_group = []
            avg_color = [0, 0, 0]
            avg_edge_number = 0
            avg_size = 0
            for cnt_dic in compare_overlap_queue:
                if cnt_dic['label'] == label_i : 
                    tmp_group.append( cnt_dic['cnt'] )
                    approx = cv2.approxPolyDP(cnt_dic['cnt'], 10, True)
                    factor = 4*np.pi*cv2.contourArea(cnt_dic['cnt'])/ float(pow(len(cnt_dic['cnt']),2))
                    if factor < 0.9:
                        avg_edge_number += len(approx)
                    
                    for i in range(3):
                        avg_color[i] += cnt_dic['color'][i]
                    
                    avg_size += cv2.contourArea(cnt_dic['cnt'])
                    
            # end compare_overlap_queue for 
            
            if len(tmp_group) < 1 :
                continue
            count = len(tmp_group)
            avg_edge_number /= count
            avg_size /= count
            for i in range(3):
                avg_color[i] /= count                        
            
            final_nonoverlap_cnt_group.append({ 'cnt':tmp_group, 'edge_number':avg_edge_number, 'color':avg_color, 'size':avg_size,'count':count })
            
        # end each label make group for
        
        #final_nonoverlap_cnt_group = CheckOverlap(final_nonoverlap_cnt_group)
       
        #contour_image[:] = BLACK
        #for cnt in final_nonoverlap_cnt_group:
            #cnt = cnt['cnt']
            #cv2.drawContours( contour_image, np.array(cnt), -1, GREEN, 2 )
        #cv2.imshow('cnt', contour_image)
        #cv2.waitKey(0)         
        
        # draw final result
        final_group_cnt = []
        contour_image = image_resi.copy()
        contour_image[:] = contour_image[:]/3.0
        
        # sort list from little to large
        final_nonoverlap_cnt_group.sort( key = lambda x: len(x['cnt']) , reverse = False)  
        
        for tmp_group in final_nonoverlap_cnt_group:
           
            if len(tmp_group) < 2 :
                continue
            
            final_group_cnt.append(tmp_group['cnt'])
            contour_image_each = image_resi.copy()
            # darken the image to make the contour visible
            contour_image_each[:] = contour_image_each[:]/3.0                
            COLOR = switchColor[ color_index % len(switchColor) ]
            color_index += 1                
            cv2.drawContours( contour_image, np.array(tmp_group['cnt']), -1, COLOR, 2 )
            cv2.drawContours( contour_image_each, np.array(tmp_group['cnt']), -1, COLOR, 2 )

            if _showImg['final_each_group_result']:            
                    cv2.imshow(fileName+' _label['+str(color_index)+']_Count['+str(tmp_group['count'])+']_size['+str(tmp_group['size'])+']_color'+str(tmp_group['color'])+'_edgeNumber['+str(tmp_group['edge_number'])+']', ShowResize(contour_image_each) )
                    cv2.waitKey(0)     
            if _writeImg['final_each_group_result']:
                    cv2.imwrite( output_path + fileName[:-4] +'_k_label['+str(color_index)+']_Count['+str(tmp_group['count'])+']_size['+str(tmp_group['size'])+']_color'+str(tmp_group['color'])+'_edgeNumber['+str(tmp_group['edge_number'])+'].jpg', contour_image_each )  

        # end final_nonoverlap_cnt_group for 

        if _evaluate:
            resize_ratio = resize_height/float(height)
            tp, fp, fn, pr, re, fm, er = Evaluate_detection_performance( image_resi, fileName, final_group_cnt, resize_ratio, evaluate_csv_path )
            #evaluation_csv = [['Image name','TP','FP','FN','Precision','Recall','F_measure','Error_rate']]
            evaluation_csv.append( [ fileName, tp, fp, fn, pr, re, fm, er ])
            
            
        
        if _record_by_csv:
            Record_by_CSV( fileName, final_group_cnt, contour_image )

        contour_image = cv2.resize( contour_image, (0,0), fx = height/resize_height, fy = height/resize_height)
        combine_image = np.concatenate((color_image_ori, contour_image), axis=1) 

        if _showImg['final_result']:
            cv2.imshow(fileName+' final_result', ShowResize(combine_image) )
            cv2.waitKey(0)     
        if _writeImg['final_result']:
            cv2.imwrite( output_path + fileName[:-4] +'_l_final_result.jpg', combine_image )            
        
        
        
        print 'Finished in ',time.time()-start_time,' s'
     
        print '-----------------------------------------------------------'    
        each_img_time = time.time() - start_time
        if each_img_time > max_time : 
            max_time = each_img_time
            max_time_img = fileName
        if each_img_time < min_time :
            min_time = each_img_time
            min_time_img = fileName
    
    if _evaluate:
        f = open(evaluate_csv_path+'evaluate-bean.csv',"wb")
        w = csv.writer(f)
        w.writerows(evaluation_csv)
        f.close()           
            
    print 'img:', max_time_img ,' max_time:',max_time,'s'
    print 'img:', min_time_img ,'min_time:',min_time,'s'
        

def Evaluate_detection_performance( img, fileName, final_group_cnt, resize_ratio, evaluate_csv_path ):

    tp = 0
    fp = 0
    fn = 0 
    pr = 0.0
    re = 0.0
    fm = 0.0
    er = 0.0
    groundtruth_list = []
    with open(evaluate_csv_path+fileName+'.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #groundtruth_list.append( { 'Group':int(row['Group']), 'X':int(int(row['X'])*resize_ratio), 'Y':int(int(row['Y'])*resize_ratio) } )
            groundtruth_list.append( { 'Group':int(row['Group']), 'X':int(row['X']), 'Y':int(row['Y']) } )  
            
    cnt_area_coordinate = Get_Cnt_Area_Coordinate(img, final_group_cnt)     
    cnt_area_coordinate.sort( key = lambda x: len(x) , reverse = False) 
    
    groundtruth_count = len(groundtruth_list)
    program_count = len(cnt_area_coordinate)
    
    #blank_img = img.copy()
    #blank_img[:] = blank_img[:]/3.0
    #for g_dic in groundtruth_list:
        #cv2.circle(blank_img,(int(g_dic['X']),int(g_dic['Y'])),2,(0,0,255),2)
   
    #print groundtruth_list[-1]['Y'], groundtruth_list[-1]['X']
    #print cnt_area_coordinate[-1][0]
    
    for g_dic in groundtruth_list:
        for cnt in cnt_area_coordinate:
            if [g_dic['Y'],g_dic['X']] in cnt :
                tp += 1
                cnt_area_coordinate.remove(cnt)
                break
    
    fp = program_count - tp
    fn = groundtruth_count - tp
    
    if tp+fp > 0:
        pr = tp / float(tp+fp)
    if tp+fn > 0:   
        re = tp / float(tp+fn)
    if pr+re > 0:
        fm = 2*pr*re / (pr+re)
    if groundtruth_count > 0:
        er = abs( program_count - groundtruth_count ) / float(groundtruth_count)
    print program_count,groundtruth_count
    return tp, fp, fn, pr, re, fm, er

def Get_Cnt_Area_Coordinate( img, final_group_cnt ):
    
    cnt_area_coordinate = []
    blank_img = np.zeros(img.shape[:2], np.uint8)
    
    for cnt_group in final_group_cnt:
        for cnt in cnt_group:
            blank_img[:] = 0
            cv2.drawContours(blank_img,[cnt],-1,255,-1)
            #cv2.imshow('blank',blank_img)
            #cv2.waitKey(0)
            # use argwhere to find all coordinate which value == 1 ( cnt area )
            cnt_area_coordinate.append( (np.argwhere(blank_img==255)).tolist() )
    
    return cnt_area_coordinate

def Search_whole_Img_by_Obvious_Cnt( image_resi, obvious_cnt_group_list ):
    
    
    #cv2.imshow('search',image_resi)
    #cv2.waitKey(100)
    #detect edge by dynamical threshold    
    gray = cv2.cvtColor( image_resi, cv2.COLOR_BGR2GRAY )

    thresh, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print 'thresh:',thresh
    
    match_cnt_group_list = []
    
    for obvious_cnt_group in obvious_cnt_group_list:
        
        obvious_cnt_group = obvious_cnt_group['group_dic']
        # use color, size and shape to match
        # find three avg parameter feature set to be a matching module
        avg_color = [0.0]*len(obvious_cnt_group[0]['color'])
        avg_shape = [0.0]*len(obvious_cnt_group[0]['shape'])
        avg_size = [0.0]*len(obvious_cnt_group[0]['size'])
        print 'find matching module'
        for obvious_cnt in obvious_cnt_group:
            
            for i in range( len(obvious_cnt['color']) ):
                avg_color[i] += obvious_cnt['color'][i]
            for i in range( len(obvious_cnt['shape']) ):
                avg_shape[i] += obvious_cnt['shape'][i]                
            for i in range( len(obvious_cnt['size']) ):
                avg_size[i] += cv2.contourArea(obvious_cnt['cnt'])       
           
        cnt_number = len(obvious_cnt_group)   
        
        for i in range( len(obvious_cnt['color']) ):
            avg_color[i] /= cnt_number
        for i in range( len(obvious_cnt['shape']) ):
            avg_shape[i] /= cnt_number        
        for i in range( len(obvious_cnt['size']) ):
            avg_size[i] /= cnt_number  
            
        matching_module = { 'cnt':[], 'shape':avg_shape, 'color':avg_color, 'size':avg_size }
        
        # find the most different cnt from matching module as bounding module  
        max_color_dis = 0.0
        max_shape_dis = 0.0
        max_size_dis = 0.0
        print 'find bounding module'
        for obvious_cnt in obvious_cnt_group:
            color_dis = Eucl_distance( obvious_cnt['color'], matching_module['color'] )
            shape_dis = Eucl_distance( obvious_cnt['shape'], matching_module['shape'] )
            size_dis = Eucl_distance( cv2.contourArea(obvious_cnt['cnt']), matching_module['size'] )
            if color_dis > max_color_dis:
                max_color_dis = color_dis
            if shape_dis > max_shape_dis:
                max_shape_dis = shape_dis                
            if size_dis > max_size_dis:
                max_size_dis = size_dis       
        
        bounding_dis = { 'shape':0.0, 'color':0.0, 'size':0.0, 'combine':0.0 }
        bounding_module = {}
        for obvious_cnt in obvious_cnt_group:
            color_dis = Eucl_distance( obvious_cnt['color'], matching_module['color'] ) 
            shape_dis = Eucl_distance( obvious_cnt['shape'], matching_module['shape'] ) 
            size_dis = Eucl_distance( cv2.contourArea(obvious_cnt['cnt']), matching_module['size'] )  
            tmp_dis = color_dis/float(max_color_dis) + shape_dis/float(max_shape_dis) + size_dis/float(max_size_dis)
            if tmp_dis > bounding_dis['combine'] :
                bounding_dis['combine'] = tmp_dis
                bounding_dis['shape'] = shape_dis
                bounding_dis['color'] = color_dis
                bounding_dis['size'] = size_dis                                   
                bounding_module = obvious_cnt
                
                
        
        
        match_dic_list = []
        for thre in np.arange( thresh, thresh+1, thresh  ):
            
            print 'thre:',thre
            edge = cv2.Canny( gray.copy(),0.5*thre,thre )          
            contour_list = cv2.findContours(edge,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]             

            c_img = image_resi.copy()
            c_img[:] = c_img[:]/3.0
            cv2.drawContours( c_img, contour_list, -1, GREEN, 1 )        
            for i in range( len(contour_list) ):
                area = cv2.contourArea(contour_list[i]) 
                shape_factor = 4*np.pi*area / float( pow(len(contour_list[i]), 2 ) )
                if shape_factor < 0.5:
                    contour_list[i] = cv2.convexHull(contour_list[i])
            cv2.drawContours( c_img, contour_list, -1, RED, 1 ) 
            cv2.imshow('img convexhul',c_img)
            cv2.waitKey(100)
            print 'extract feature'
            c_list, cnt_shape_list, cnt_color_list, cnt_size_list, cnt_color_gradient_list = get_contour_feature.extract_feature( image_resi, contour_list )
            
            cnt_dic_list = []
            for i in range( len(c_list) ):
                cnt_dic_list.append( { 'cnt':c_list[i], 'shape':cnt_shape_list[i], 'color':cnt_color_list[i], 'size':cnt_size_list[i], 'color_gradient':cnt_color_gradient_list[i] } )            
            print 'matching'    
            # matching
            for cnt_dic in cnt_dic_list:
                if Eucl_distance( cnt_dic['shape'], matching_module['shape'] ) <= bounding_dis['shape']*1.5 and Eucl_distance( cv2.contourArea(cnt_dic['cnt']), matching_module['size'] ) <= bounding_dis['size']*1.5 : 
                #if Eucl_distance( cnt_dic['shape'], matching_module['shape'] ) <= bounding_dis['shape']*1.5 or Eucl_distance( cnt_dic['color'], matching_module['color'] ) <= bounding_dis['color']*1.5 or Eucl_distance( cnt_dic['size'], matching_module['size'] ) <= bounding_dis['size']*1.5 : 
                    print 'shape: ',Eucl_distance( cnt_dic['shape'], matching_module['shape'] ), bounding_dis['shape']
                    print 'color: ',Eucl_distance( cnt_dic['color'], matching_module['color'] ) , bounding_dis['color'],cnt_dic['color'], matching_module['color']
                    print 'size:',Eucl_distance( cv2.contourArea(cnt_dic['cnt']), matching_module['size'] ) , bounding_dis['size'],cv2.contourArea(cnt_dic['cnt']),matching_module['size']
                    print '----------------------------------------------------------------------------------------------'
                    match_dic_list.append(cnt_dic)
                    
        # end dynamical edge threshold for 
         
        contour_image_obvious = image_resi.copy()
        contour_image_obvious[:] = contour_image_obvious[:]/3.0
        print 'draw'
        for obvious_cnt in obvious_cnt_group:
            cv2.drawContours( contour_image_obvious, [obvious_cnt['cnt']], -1, GREEN, 2 )
        
        contour_image_match = image_resi.copy()
        contour_image_match[:] = contour_image_match[:]/3.0        
        color_index = 0
        for match_cnt in match_dic_list:
            color_index = (color_index+1) % len(switchColor)
            COLOR = switchColor[color_index]
            cv2.drawContours( contour_image_match, [match_cnt['cnt']], -1, COLOR, 1 )        
        
        combine_image = np.concatenate( (image_resi,contour_image_obvious),axis=1 ) 
        combine_image = np.concatenate( (combine_image,contour_image_match),axis=1 ) 
        
        cv2.imshow('find match cnt',combine_image)
        cv2.waitKey(100)
        
        match_cnt_group_list.append( match_dic_list )
    
    # end obvious cnt group for
    return match_cnt_group_list
    
def Avg_Img_Gredient( img, model = 'lab' ):
    
    kernel = np.array( [[-1,-1,-1],
                        [-1, 8,-1],
                        [-1,-1,-1]]  )    
    
    if model == 'lab' :
        
        height, width = img.shape[:2]       
        lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
        lab_l = lab[:,:,0]
        lab_a = lab[:,:,1]
        lab_b = lab[:,:,2]      
     
        lab_list = [ lab_l, lab_a, lab_b ]
        gradient_list = []
        
        for lab_channel in lab_list :
            gradient = cv2.filter2D(lab_channel, -1, kernel)
            gradient_list.append(gradient)
        
        avg_gradient = 0.0  
        for x in range(height):
            for y in range(width):
                avg_gradient += math.sqrt( pow( gradient_list[0][x,y],2 ) +  pow( gradient_list[1][x,y],2 ) + pow( gradient_list[2][x,y],2 ) )  
                
        avg_gradient /=  ( float(height) * float(width) )
                
    return avg_gradient       
                


# unused func
def Record_by_CSV( filename, cnt_list, contour_image ):
    
    coordinar_list = [ [ 'Group','Y','X' ] ]
    img = contour_image.copy()
    #img[:]=BLACK
    # for each group
    for group_i in range( len(cnt_list) ):
        for cnt in cnt_list[group_i]:
            x, y = GetMoment(cnt)
            coordinar_list.append( [ group_i, int(y), int(x) ] )
            cv2.circle(img,(int(y),int(x)),2,(0,0,255),2)
    #cv2.imshow('coordinate: '+str(x)+','+str(y),img)
    #cv2.waitKey(100)
    f = open(csv_output+filename[:-4]+'.csv',"wb")
    w = csv.writer(f)
    w.writerows(coordinar_list)
    f.close()       
    
def Sharpen(img):
    
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0  
   
    return cv2.filter2D(img, -1, kernel_sharpen) 

def Eucl_distance(a,b):
    
    if type(a) != np.ndarray :
        a = np.array(a)
    if type(b) != np.ndarray :
        b = np.array(b)
    
    return np.linalg.norm(a-b) 

# unused func
def Draw_image( image_resi, c_list, label_list, max_label ):
    
    if type(label_list) != np.ndarray :
        label_list = np.array(label_list)
    
    samples_mask = np.zeros_like(c_list, dtype=bool)
    samples_mask[:] = True     
    index_mask = ( label_list == max_label )
    image = image_resi.copy()
    contour_image = np.zeros(image_resi.shape, np.uint8)
    contour_image[:] = BLACK            

    c_list = np.array(c_list)
    for c in c_list[samples_mask ^ index_mask]  : 
        if len(c_list) > 1 :
            cv2.drawContours( contour_image, [c], -1, GREEN, 1 )
        else:
            cv2.drawContours( contour_image, c_list, -1, GREEN, 1 )

    tmp_c = []
    drawed_list = []
    max_contour_list = list(c_list[samples_mask & index_mask])
    max_contour_list.sort( key = lambda x: len(x) , reverse = False) 
    
    #print 'len(max_contour_list):',len(max_contour_list)
    count_loss = 0
    for c in max_contour_list  :  
        if len(max_contour_list) > 1 :
            #if IsOverlapAll( c, drawed_list ) :
                #count_loss += 1
                #tmp_c = c
                #continue               
            cv2.drawContours( contour_image, [c], -1, RED, 1 )                      
            cv2.drawContours( image, [c], -1, RED, 1 )
            drawed_list.append(c)
            tmp_c = c
        else:
            cv2.drawContours( contour_image, max_contour_list, -1, RED, 1 )
            cv2.drawContours( image, max_contour_list, -1, RED, 1 )                    
        
    combine_image = np.concatenate((image, contour_image), axis=1)      
    return combine_image, count_loss

def CheckOverlap( cnt_dic_list, keep = 'keep_inner' ):
    
    if cnt_dic_list == []:
        return []
    
    checked_list = []
    
    if keep == 'group_weight':
        
        label_list = [ x['label'] for x in cnt_dic_list ]
        label_change_list = []
        label_group_change = []
        label_change_dic = {}  
        
        for cnt_i in range( len(cnt_dic_list)-1 ):
            for cnt_k in range( cnt_i+1, len(cnt_dic_list) ):
                
                if cnt_dic_list[cnt_i]['group_weight'] > 0 and cnt_dic_list[cnt_k]['group_weight'] > 0 :
                    if IsOverlap(cnt_dic_list[cnt_i]['cnt'], cnt_dic_list[cnt_k]['cnt']):
                        
                        if cnt_dic_list[cnt_i]['group_weight'] > cnt_dic_list[cnt_k]['group_weight']:
                            cnt_dic_list[cnt_k]['group_weight'] = 0
                            label_change_list.append( (cnt_dic_list[cnt_k]['label'],cnt_dic_list[cnt_i]['label']) )
                        else:
                            cnt_dic_list[cnt_i]['group_weight'] = 0
                            label_change_list.append( (cnt_dic_list[cnt_i]['label'],cnt_dic_list[cnt_k]['label']) )
     
        
        # check if overlap contours are same contour , if true makes them same label
        for label_change in set(label_change_list):
            if label_change_list.count(label_change) >= 0.5*label_list.count(label_change[0]):
                found = False
                for label_group_i in range( len(label_group_change) ):
                    if label_change[0] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[1])
                    elif label_change[1] in label_group_change[label_group_i]:
                        found = True
                        label_group_change[label_group_i].append(label_change[0])
                
                if not found : 
                    label_group_change.append([label_change[0],label_change[1]])
                    
                #label_change_dic[label_change[0]] = label_change[1]

        for label_group in label_group_change:
            for label in label_group:
                label_change_dic[label] = label_group[0]
        
        for cnt_dic in cnt_dic_list:
            if cnt_dic['group_weight'] > 0:
                if cnt_dic['label'] in label_change_dic:
                    cnt_dic['label'] = label_change_dic[cnt_dic['label']]
                checked_list.append(cnt_dic)
        
    else:
        
        if keep == 'keep_inner':
            # sort list from little to large
            cnt_dic_list.sort( key = lambda x: len(x['cnt']) , reverse = False)
            
        elif keep == 'keep_outer':
            cnt_dic_list.sort( key = lambda x: len(x['cnt']) , reverse = True)
        
        for c_dic in cnt_dic_list  : 
            if IsOverlapAll( c_dic, checked_list ) : 
                continue               
            checked_list.append(c_dic) 
    
    # end keep if
    
    return checked_list
        
def IsOverlap( cnt1, cnt2 ):
    
    if cnt1 == [] or cnt2 == [] :
        return False
    
    c1M = GetMoment(cnt1)
    c2M = GetMoment(cnt2)
    c1_min_d = MinDistance(cnt1)
    c2_min_d = MinDistance(cnt2)
    moment_d = Eucl_distance( c1M, c2M )
    
    if min(c1_min_d,c2_min_d) == 0:
        return False
    
    return ( moment_d < c1_min_d or moment_d < c2_min_d ) and max(c1_min_d,c2_min_d)/min(c1_min_d,c2_min_d) <= 3

def IsOverlapAll( cnt_dic, cnt_dic_list ):
    
    if cnt_dic == [] or len(cnt_dic_list) < 1 :
        return False

    for c_dic in cnt_dic_list :
        #if len(c) == len(cnt) and GetMoment(c) == GetMoment(cnt):
            ##print 'same one'
            #continue
        if IsOverlap( cnt_dic['cnt'], c_dic['cnt'] ) :
            return True
    
    return False

def SplitColorChannel( img ):
    
    bgr_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ) 
    #bgr_gray = cv2.GaussianBlur(bgr_gray, (3, 3), 0)  
    thresh_bgr_gray = cv2.threshold(bgr_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]    
    
    
    bgr_b = img[:,:,0]
    bgr_g = img[:,:,1]
    bgr_r = img[:,:,2]   
    bgr_b = cv2.GaussianBlur(bgr_b, (5, 5), 0)
    bgr_g = cv2.GaussianBlur(bgr_g, (5, 5), 0)
    bgr_r = cv2.GaussianBlur(bgr_r, (5, 5), 0)
    thresh_bgr_b = cv2.threshold(bgr_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_bgr_g = cv2.threshold(bgr_g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_bgr_r = cv2.threshold(bgr_r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]    
    
    hsv = cv2.cvtColor( img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    hsv_h = hsv[:,:,0]
    hsv_s = hsv[:,:,1]
    hsv_v = hsv[:,:,2]   
    hsv_h = cv2.GaussianBlur(hsv_h, (5, 5), 0)
    hsv_s = cv2.GaussianBlur(hsv_s, (5, 5), 0)
    hsv_v = cv2.GaussianBlur(hsv_v, (5, 5), 0)
    thresh_hsv_h = cv2.threshold(hsv_h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_hsv_s = cv2.threshold(hsv_s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_hsv_v = cv2.threshold(hsv_v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]                           

    lab = cv2.cvtColor( img, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (5,5), 0)
    lab_l = lab[:,:,0]
    lab_a = lab[:,:,1]
    lab_b = lab[:,:,2]
    lab_l = cv2.GaussianBlur(lab_l, (5, 5), 0)
    lab_a = cv2.GaussianBlur(lab_a, (5, 5), 0)
    lab_b = cv2.GaussianBlur(lab_b, (5, 5), 0)  
    thresh_lab_l = cv2.threshold(lab_l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_lab_a = cv2.threshold(lab_a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    thresh_lab_b = cv2.threshold(lab_b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]   
    
    return { 'img_bgr_gray':bgr_gray, 'img_bgr':img, 'img_b':bgr_b, 'img_g':bgr_g, 'img_r':bgr_r, 'thre_bgr_gray':thresh_bgr_gray, 'thre_b':thresh_bgr_b, 'thre_g':thresh_bgr_g, 'thre_r':thresh_bgr_r 
             },{ 'img_hsv':hsv, 'img_h':hsv_h, 'img_s':hsv_s, 'img_v':hsv_v, 'thre_h':thresh_hsv_h, 'thre_s':thresh_hsv_s, 'thre_v':thresh_hsv_v 
                 },{ 'img_lab':lab, 'img_l':lab_l, 'img_a':lab_a, 'img_b':lab_b, 'thre_l':thresh_lab_l, 'thre_a':thresh_lab_a, 'thre_b':thresh_lab_b }

def ShowResize( img ):
    
    h, w = img.shape[:2]
    
    if _show_resize[1] == 'height':
        ratio = _show_resize[0] / float(h)
    else :
        ratio = _show_resize[0] / float(w)
    
    return cv2.resize( img, (0,0), fx = ratio, fy = ratio )
        
def MinDistance(cnt):
    
    cM = GetMoment(cnt)
    if len(cnt[0][0]) == 1:
        cnt = cnt[0]
    min_d = Eucl_distance( (cnt[0][0][0],cnt[0][0][1]), cM )
    for c in cnt :
        d = Eucl_distance( (c[0][0],c[0][1]), cM ) 
        if d < min_d :
            min_d = d
            
    return min_d

def LAB2Gray(img):
    
    _w, _h, _c = img.shape
    
    gray = np.zeros(img.shape[:2], np.uint8) 

    for i in xrange(_w):
        for k in xrange(_h):
            a = int(img[i][k][1])
            b = int(img[i][k][2])
            gray[i,k] = ( a + b )/2
            
            
    return gray
    
def GetMoment(cnt):
    
    num = len(cnt)
    if num < 2 :
        return cnt
    cx = 0
    cy = 0
    for c in cnt :
        if isinstance(c[0][0], np.ndarray):
            c = c[0]
        cx += float(c[0][0])
        cy += float(c[0][1])
        
    return float(cx)/num, float(cy)/num

def Hierarchical_clustering( feature_list, fileName, para, edge_type, cut_method = 'elbow' ):

    if len(feature_list) < 2:
        return [0]*len(feature_list)
    
    all_same = True
    for feature in feature_list:
        if feature != feature_list[0]:
            all_same = False
            break
    
    if all_same:
        print 'all in one group!'
        return [0]*len(feature_list)   
    
    # hierarchically link cnt by order of distance from distance method 'ward'
    #print feature_list
    cnt_hierarchy = linkage( feature_list, 'ward')
    #cnt_hierarchy = linkage( feature_list)
    
    max_d = 10
    if cut_method == 'elbow' or True:
        last = cnt_hierarchy[:, 2]
        #print 'last:',last
        last = [ x for x in last if x > 0 ]
        #print 'last:',last
        acceleration = np.diff(last) 
        
        #acceleration = map(abs, np.diff(acceleration) )
        
        #acceleration_rev = acceleration[::-1]
        #print 'acceleration:',acceleration 
        
        if len(acceleration) < 2 :
            return [0]*len(feature_list)
        avg_diff = sum(acceleration)/float(len(acceleration))
        tmp = acceleration[0]
        
        avg_list = [x for x in acceleration if x > avg_diff]
        avg_diff = sum(avg_list)/float(len(avg_list))
        
        off_set = 5
        
        rario = []
        cut_point_list = []
        for i in xrange( 1,len(acceleration) ):
       
            if acceleration[i] > avg_diff:
                #cut_point_list.append( [ i, acceleration[i]/(tmp/float(i) ) ] )
                
                tmp_offset_prev = off_set
                prev = i - off_set
                if prev < 0 :
                    prev = 0
                    tmp_offset_prev = i-prev
                rario.append(acceleration[i]/( sum(acceleration[prev:i]) / float(tmp_offset_prev) ))
                cut_point_list.append( [ i, acceleration[i]/( sum(acceleration[prev:i]) / float(tmp_offset_prev) ) ] )
                #cut_point_list.append( [ i, acceleration[i] ] )
                #print 'i:',i+1,' ratio:',acceleration[i]/( sum(acceleration[n:i]) / float(off_set) )
                
            tmp += acceleration[i]
            
        if len(cut_point_list) < 1 :
            print 'all in one group!'
            return [0]*len(feature_list)     
        
        cut_point_list.sort( key = lambda x : x[1], reverse = True )
        
        #print 'cut index:',cut_point_list[0][0]+1,' diff len:',len(acceleration)
        max_d = last[cut_point_list[0][0]]
        max_ratio = cut_point_list[0][1]
        
        if max_ratio < 2.0 :
            print 'all in one group! max_ratio:',max_ratio
            return [0]*len(feature_list)  
        
        #max_d = last[acceleration.argmax()]
    #elif cut_method == 'inconsistency':
    
    #plt.bar(left=range(len(rario)),height=rario)  
    plt.bar(left=range(len(acceleration)),height=acceleration)   
    plt.title( para+' cut_point : '+str(cut_point_list[0][0]+1)+'  | value: '+str(acceleration[cut_point_list[0][0]])+' | ratio: '+ str(max_ratio)  )
    
    if _showImg['cluster_histogram']:
        plt.show()
    if _writeImg['cluster_histogram']:
        plt.savefig(output_path+fileName[:-4]+'_f_para['+para+']_his['+str(edge_type)+'].png')    
    plt.close()    
    
    #print 'acceleration.argmax():',acceleration.argmax()
    clusters = fcluster(cnt_hierarchy, max_d, criterion='distance')   
    print '----------------------------------'
    return clusters
    

if __name__ == '__main__' :
    
    t_start_time = time.time()
   
    main()
    #_local = False
    ##output_path = './output_global/'    
    #main()
    print 'All finished in ',time.time()-t_start_time,' s'
