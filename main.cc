#include "stdio.h"
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "ipa_room_segmentation/voronoi_segmentation.h"
#include "ipa_room_segmentation/meanshift2d.h"
#include "ipa_room_segmentation/timer.h"


typedef enum		   // 地图状态：
{
    UNKNOWN = 0,	   // 未知
    CLEANED = 1,	   // 已清扫
    EMPTY   = 2,	   // 空地
    BAR		  = 3,	   // 障碍物
}MapStatusEnum;


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

//函数功能：排序(从小到大)
template<typename T>
void quick_sort(vector<T>& arr, int left, int right, vector<int>& arr_index)
{
    int i, j;
    int t1,temp1;
    T temp,t;

    if (left>right)
        return;

    temp = arr[left];
    temp1 = arr_index[left];
    i = left;
    j = right;

    while (i != j)
    {
        while (arr[j] >= temp && i<j)
            j--;
        while (arr[i] <= temp && i<j)
            i++;
        if (i<j)
        {
            t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
            t1 = arr_index[i];
            arr_index[i] = arr_index[j];
            arr_index[j] = t1;
        }
    }
    arr[left] = arr[i];
    arr[i] = temp;

    arr_index[left] = arr_index[i];
    arr_index[i] = temp1;
    quick_sort(arr,left, i - 1,arr_index);
    quick_sort(arr,i + 1, right,arr_index);
}

//函数功能：根据给定索引进行排序
template<typename T>
void resort_arr(vector<T>& arr, vector<int> indexs)
{
    int len = arr.size();
    T* backup = new T[len];
    for(int i=0; i<len; i++){
        backup[i] = arr[i];
    }
    for(int i=0; i<len; i++){
        arr[i] = backup[indexs[i]];
    }
}

/*
函数功能：在二维数组中绘制指定位置的线段
arr：二维数组，可理解为二值化后的灰度图像
cols：图像的宽
ltop：线段左上位置的端点
rbottom：线段的右下位置的端点
val：绘制直线线段的值
*/
void set_array(uchar* arr, int cols, Point ltop, Point rbottom, int val)
{
    // 直线绘制
    if(ltop.x == rbottom.x || ltop.y == rbottom.y)
    {
        for(int i=ltop.y; i<=rbottom.y; i++)
        {
            for(int j=ltop.x; j<=rbottom.x; j++)
            {
                *(arr+i*cols+j) = val;
            }
        }
    }
    // 斜线绘制,绘制斜线时须在X和Y两个方向进行遍历,否则对于近乎水平或竖直的斜线绘制会有问题
    else
    {
        float x_slope = float(rbottom.y - ltop.y) / (rbottom.x - ltop.x);
        for(int i=ltop.x; i<=rbottom.x; i++)
        {
            int yoffset = (i-ltop.x)*x_slope + ltop.y;
            *(arr+yoffset*cols+i) = val;
        }
        float y_slope = float(rbottom.x - ltop.x) / (rbottom.y - ltop.y);
        for(int j=ltop.y; j<=rbottom.y; j++)
        {
            int xoffset = (j-ltop.y)*y_slope + ltop.x;
            *(arr+j*cols+xoffset) = val;
        }
    }
}

/*
函数功能：在竖直方向和水平方向合并邻近的直线线段(对斜线不作处理)
image_data：二维数组，可以认为是二值化后的灰度图像
img_w：图像宽
img_h：图像高
merge_step：每merge_step行或列作为一个线段合并的计算单元
merge_max_gap：前后线段之间的首尾距离小于merge_max_gap才满足合并条件
merge_len_thresh：合并的两条线段的长度和大于merge_len_thresh才满足合并条件(避免斜线被合并)
*/
void merge_lines(uchar* image_data,
                 int img_w,
                 int img_h,
                 int merge_step = 3,
                 int merge_max_gap = 3,
                 int merge_len_thresh = 30)
{
    // 竖直方向合并线段
    for(int n=0; n<img_w-merge_step; n++)
    {
        int line_num = 0; //检测到的线段数量
        vector<vector<int>> vec_lines; //存储计算单元内每列的线段首尾偏移数据
        for(int x=n; x<n+merge_step; x++)
        {
            vector<int> vec_line;
            int last_val = 0;
            int head_off = 0;
            int tail_off = 0;
            for(int y=0; y<img_h; y++)
            {
                int gray_scale = *(image_data+y*img_w+x);
                if(gray_scale==255 && last_val==0){
                    head_off = y;
                }
                if((gray_scale==0 && last_val==255)){
                    tail_off = y;
                    // 长度为1的线段忽略，避免处理后直角处出现断裂
                    if(tail_off - head_off == 1)
                    {
                        last_val = 0;
                        continue;
                    }
                    vec_line.push_back(head_off);
                    vec_line.push_back(tail_off);
                    line_num++;
                }
                last_val = gray_scale;
            }
            vec_lines.push_back(vec_line);
        }
        // 不少于2条线段执行合并操作
        if(line_num >= 2)
        {
            vector<int> heads; //保存线段的头部偏移
            vector<int> tails; //保存线段的尾部偏移
            vector<int> indexs;
            for(int i=0; i<vec_lines.size(); i++)
            {
                for(int j=0; j<vec_lines[i].size(); j+=2)
                {
                    heads.push_back(vec_lines[i][j]);
                    tails.push_back(vec_lines[i][j+1]);
                    indexs.push_back(indexs.size());
                }
            }

            // 对计算单元内所有的线段头部偏移进行从小到大的排序
            quick_sort(heads, 0, line_num-1, indexs);
            // 所有的尾部偏移根据头部偏移的排序索引重新排序
            resort_arr(tails, indexs);

            vector<int> merge_heads; //线段合并后的头部偏移数组,同一列需要合并的线段可能不止一条
            vector<int> merge_tails; //线段合并后的尾部偏移数组
            int current_merge_head = heads[0]; //当前合并线段的头部偏移
            int current_merge_tail = tails[0]; //当前合并线段的尾部偏移
            for(int i=1; i<line_num; i++)
            {
                int merge_len = current_merge_tail - current_merge_head;
                int line_len = tails[i] - heads[i];
                // 合并条件：当前合并线段的尾部偏移和下一条线段的头部偏移最大的距离不超过阈值，
                // 且二者长度和需大于阈值，防止邻近的两条斜线被误合并
                if(current_merge_tail-heads[i] > -merge_max_gap
                        && merge_len+line_len > merge_len_thresh)
                {
                    current_merge_tail = max(current_merge_tail, tails[i]);
                }
                else
                {
                    if(merge_len > merge_len_thresh)
                    {
                        merge_heads.push_back(current_merge_head);
                        merge_tails.push_back(current_merge_tail);
                    }
                    current_merge_head = heads[i];
                    current_merge_tail = tails[i];
                }
            }
            // 若最后一条线段也需要合并，则需要手动把合并线段加进来(此处可能会添加重复，不过不影响)
            if(current_merge_tail - current_merge_head > merge_len_thresh)
            {
                merge_heads.push_back(current_merge_head);
                merge_tails.push_back(current_merge_tail);
            }

            // 遍历计算单元里所有线段，将组成合并线段的各个子线段清除
            for(int k=0; k<merge_heads.size(); k++)
            {
                Point merge_head_pt(-1);
                Point merge_tail_pt(-1);
                for(int i=0; i<vec_lines.size(); i++)
                {
                    for(int j=0; j<vec_lines[i].size(); j+=2)
                    {
                        if(vec_lines[i][j] >= merge_heads[k] && vec_lines[i][j+1] <= merge_tails[k])
                        {
                            // 记录合并线段的头坐标点
                            if(vec_lines[i][j] == merge_heads[k] && merge_head_pt.x == -1)
                            {
                                merge_head_pt.x = n+i;
                                merge_head_pt.y = merge_heads[k];
                            }
                            // 记录合并线段的尾坐标点
                            if(vec_lines[i][j+1] == merge_tails[k] && merge_tail_pt.x == -1)
                            {
                                merge_tail_pt.x = n+i;
                                merge_tail_pt.y = merge_tails[k]-1;
                            }
                            // 将子线段的像素值置0
                            set_array(image_data, img_w, Point(n+i,vec_lines[i][j]), Point(n+i, vec_lines[i][j+1]-1), 0);
                        }
                    }
                }
                // 将合并线段的像素值置255
                set_array(image_data, img_w, merge_head_pt, merge_tail_pt, 255);
            }
        }
    }

    // 水平方向合并线段（同竖直方向合并线段同理）
    for(int n=0; n<img_h-merge_step; n++)
    {
        int line_num = 0;
        vector<vector<int>> vec_lines;
        for(int y=n; y<n+merge_step; y++)
        {
            vector<int> vec_line;
            int last_val = 0;
            int head_off = 0;
            int tail_off = 0;
            for(int x=0; x<img_w; x++)
            {
                int gray_scale = *(image_data+y*img_w+x);
                if(gray_scale==255 && last_val==0){
                    head_off = x;
                }
                if((gray_scale==0 && last_val==255)){
                    tail_off = x;
                    if(tail_off - head_off == 1)
                    {
                        last_val = 0;
                        continue;
                    }
                    vec_line.push_back(head_off);
                    vec_line.push_back(tail_off);
                    line_num++;
                }
                last_val = gray_scale;
            }
            vec_lines.push_back(vec_line);
        }

        if(line_num >= 2)
        {
            vector<int> heads;
            vector<int> tails;
            vector<int> indexs;
            for(int i=0; i<vec_lines.size(); i++)
            {
                for(int j=0; j<vec_lines[i].size(); j+=2)
                {
                    heads.push_back(vec_lines[i][j]);
                    tails.push_back(vec_lines[i][j+1]);
                    indexs.push_back(indexs.size());
                }
            }

            quick_sort(heads, 0, line_num-1, indexs);
            resort_arr(tails, indexs);

            vector<int> merge_heads;
            vector<int> merge_tails;
            int current_merge_head = heads[0];
            int current_merge_tail = tails[0];
            for(int i=1; i<line_num; i++)
            {
                int merge_len = current_merge_tail - current_merge_head;
                int line_len = tails[i] - heads[i];
                if(current_merge_tail-heads[i] > -merge_max_gap
                        && merge_len+line_len > merge_len_thresh)
                {
                    current_merge_tail = max(current_merge_tail, tails[i]);
                }
                else
                {
                    if(merge_len > merge_len_thresh)
                    {
                        merge_heads.push_back(current_merge_head);
                        merge_tails.push_back(current_merge_tail);
                    }
                    current_merge_head = heads[i];
                    current_merge_tail = tails[i];
                }
            }

            if(current_merge_tail - current_merge_head > merge_len_thresh)
            {
                merge_heads.push_back(current_merge_head);
                merge_tails.push_back(current_merge_tail);
            }

            for(int k=0; k<merge_heads.size(); k++)
            {
                Point merge_head_pt(-1,-1);
                Point merge_tail_pt(-1,-1);
                for(int i=0; i<vec_lines.size(); i++)
                {
                    for(int j=0; j<vec_lines[i].size(); j+=2)
                    {
                        if(vec_lines[i][j] >= merge_heads[k] && vec_lines[i][j+1] <= merge_tails[k])
                        {
                            if(vec_lines[i][j] == merge_heads[k] && merge_head_pt.x == -1)
                            {
                                merge_head_pt.x = merge_heads[k];
                                merge_head_pt.y = n+i;
                            }
                            if(vec_lines[i][j+1] == merge_tails[k] && merge_tail_pt.x == -1)
                            {
                                merge_tail_pt.x = merge_tails[k]-1;
                                merge_tail_pt.y = n+i;
                            }

                            set_array(image_data, img_w, Point(vec_lines[i][j], n+i), Point(vec_lines[i][j+1]-1, n+i), 0);
                        }
                    }
                }
                set_array(image_data, img_w, merge_head_pt, merge_tail_pt, 255);
            }
        }
    }
}


void show_segment_color_map(cv::Mat segmented_map)
{
    std::map<int, size_t> label_vector_index_codebook; // maps each room label to a position in the rooms vector
    size_t vector_index = 0;
    for (int v = 0; v < segmented_map.rows; ++v)
    {
        for (int u = 0; u < segmented_map.cols; ++u)
        {
            const int label = segmented_map.at<int>(v, u);
            if (label > 0 && label < 65280) // do not count walls/obstacles or free space as label
            {
                if (label_vector_index_codebook.find(label) == label_vector_index_codebook.end())
                {
                    label_vector_index_codebook[label] = vector_index;
                    vector_index++;
                }
            }
        }
    }
    //min/max y/x-values vector for each room. Initialized with extreme values
    //!求每个区域的最大外接矩形
    std::vector<int> min_x_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_x_value_of_the_room(label_vector_index_codebook.size(), 0);
    std::vector<int> min_y_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_y_value_of_the_room(label_vector_index_codebook.size(), 0);
    //vector of the central Point for each room, initially filled with Points out of the map
    std::vector<int> room_centers_x_values(label_vector_index_codebook.size(), -1);
    std::vector<int> room_centers_y_values(label_vector_index_codebook.size(), -1);
    //***********************Find min/max x and y coordinate and center of each found room********************
    //check y/x-value for every Pixel and make the larger/smaller value to the current value of the room
    for (int y = 0; y < segmented_map.rows; ++y)
    {
        for (int x = 0; x < segmented_map.cols; ++x)
        {
            const int label = segmented_map.at<int>(y, x);
            if (label > 0 && label < 65280) //if Pixel is white or black it is no room --> doesn't need to be checked
            {
                const int index = label_vector_index_codebook[label];
                min_x_value_of_the_room[index] = std::min(x, min_x_value_of_the_room[index]);
                max_x_value_of_the_room[index] = std::max(x, max_x_value_of_the_room[index]);
                max_y_value_of_the_room[index] = std::max(y, max_y_value_of_the_room[index]);
                min_y_value_of_the_room[index] = std::min(y, min_y_value_of_the_room[index]);
            }
        }
    }
    //get centers for each room
    for (size_t idx = 0; idx < room_centers_x_values.size(); ++idx)
    {
        if (max_x_value_of_the_room[idx] != 0 && max_y_value_of_the_room[idx] != 0 && min_x_value_of_the_room[idx] != 100000000 && min_y_value_of_the_room[idx] != 100000000)
        {
            room_centers_x_values[idx] = (min_x_value_of_the_room[idx] + max_x_value_of_the_room[idx]) / 2;
            room_centers_y_values[idx] = (min_y_value_of_the_room[idx] + max_y_value_of_the_room[idx]) / 2;
            cv::circle(segmented_map, cv::Point(room_centers_x_values[idx], room_centers_y_values[idx]), 2, cv::Scalar(200*256), CV_FILLED);
        }
    }
    // use distance transform and mean shift to find good room centers that are reachable by the robot
    // first check whether a robot radius shall be applied to obstacles in order to exclude room center points that are not reachable by the robot
//    cv::Mat segmented_map_copy = segmented_map;
//    cv::Mat connection_to_other_rooms = cv::Mat::zeros(segmented_map.rows, segmented_map.cols, CV_8UC1);	// stores for each pixel whether a path to another rooms exists for a robot of size robot_radius

//    goal.input_map = labeling;
//    goal.map_origin.position.x = 0;
//    goal.map_origin.position.y = 0;
//    double map_resolution = 0.05;
//    goal.return_format_in_meter = false;
//    goal.return_format_in_pixel = true;
//    double robot_radius = 0.4;
//    if (robot_radius > 0.0)
//    {
//        // consider robot radius for exclusion of non-reachable points
//        segmented_map_copy = segmented_map.clone();
//        cv::Mat map_8u, eroded_map;
//        segmented_map_copy.convertTo(map_8u, CV_8UC1, 1., 0.);
//        int number_of_erosions = (robot_radius / map_resolution);
//        cv::erode(map_8u, eroded_map, cv::Mat(), cv::Point(-1, -1), number_of_erosions);
//        for (int v=0; v<segmented_map_copy.rows; ++v)
//            for (int u=0; u<segmented_map_copy.cols; ++u)
//                if (eroded_map.at<uchar>(v,u) == 0)
//                    segmented_map_copy.at<int>(v,u) = 0;

//        // compute connectivity of remaining accessible room cells to other rooms
//        bool stop = false;
//        while (stop == false)
//        {
//            stop = true;
//            for (int v=1; v<segmented_map_copy.rows-1; ++v)
//            {
//                for (int u=1; u<segmented_map_copy.cols-1; ++u)
//                {
//                    // skip already identified cells
//                    if (connection_to_other_rooms.at<uchar>(v,u) != 0)
//                        continue;

//                    // only consider cells labeled as a room
//                    const int label = segmented_map_copy.at<int>(v,u);
//                    if (label <= 0 || label >= 65280)
//                        continue;

//                    for (int dv=-1; dv<=1; ++dv)
//                    {
//                        for (int du=-1; du<=1; ++du)
//                        {
//                            if (dv==0 && du==0)
//                                continue;
//                            const int neighbor_label = segmented_map_copy.at<int>(v+dv,u+du);
//                            if (neighbor_label>0 && neighbor_label<65280 && (neighbor_label!=label || (neighbor_label==label && connection_to_other_rooms.at<uchar>(v+dv,u+du)==255)))
//                            {
//                                // either the room cell has a direct border to a different room or the room cell has a neighbor from the same room label with a connecting path to another room
//                                connection_to_other_rooms.at<uchar>(v,u) = 255;
//                                stop = false;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    // compute the room centers
//    MeanShift2D ms;
//    for (std::map<int, size_t>::iterator it = label_vector_index_codebook.begin(); it != label_vector_index_codebook.end(); ++it)
//    {
//        int trial = 1; 	// use robot_radius to avoid room centers that are not accessible by a robot with a given radius
//        if (robot_radius <= 0.)
//            trial = 2;

//        for (; trial <= 2; ++trial)
//        {
//            // compute distance transform for each room on the room cells that have some connection to another room (trial 1) or just on all cells of that room (trial 2)
//            const int label = it->first;
//            int number_room_pixels = 0;
//            cv::Mat room = cv::Mat::zeros(segmented_map_copy.rows, segmented_map_copy.cols, CV_8UC1);
//            for (int v = 0; v < segmented_map_copy.rows; ++v)
//                for (int u = 0; u < segmented_map_copy.cols; ++u)
//                    if (segmented_map_copy.at<int>(v, u) == label && (trial==2 || connection_to_other_rooms.at<uchar>(v,u)==255))
//                    {
//                        room.at<uchar>(v, u) = 255;
//                        ++number_room_pixels;
//                    }
//            if (number_room_pixels == 0)
//                continue;
//            cv::Mat distance_map; //variable for the distance-transformed map, type: CV_32FC1
//            cv::distanceTransform(room, distance_map, CV_DIST_L2, 5);
//            // find point set with largest distance to obstacles
//            double min_val = 0., max_val = 0.;
//            cv::minMaxLoc(distance_map, &min_val, &max_val);
//            std::vector<cv::Vec2d> room_cells;
//            for (int v = 0; v < distance_map.rows; ++v)
//                for (int u = 0; u < distance_map.cols; ++u)
//                    if (distance_map.at<float>(v, u) > max_val * 0.95f)
//                        room_cells.push_back(cv::Vec2d(u, v));
//            if (room_cells.size()==0)
//                continue;
//            // use meanshift to find the modes in that set
//            cv::Vec2d room_center = ms.findRoomCenter(room, room_cells, map_resolution);
//            const int index = it->second;
//            room_centers_x_values[index] = room_center[0];
//            room_centers_y_values[index] = room_center[1];

//            if (room_cells.size() > 0)
//                break;
//        }
//    }
    cv::Mat indexed_map = segmented_map.clone();
    for (int y = 0; y < segmented_map.rows; ++y)
    {
        for (int x = 0; x < segmented_map.cols; ++x)
        {
            const int label = segmented_map.at<int>(y,x);
            if (label > 0 && label < 65280)
                indexed_map.at<int>(y,x) = label_vector_index_codebook[label]+1;//start value from 1 --> 0 is reserved for obstacles
        }
    }
    bool display_segmented_map_=true;
    if (display_segmented_map_ == true)
    {
        // colorize the segmented map with the indices of the room_center vector
        cv::Mat color_segmented_map = indexed_map.clone();
        color_segmented_map.convertTo(color_segmented_map, CV_8U);
        cv::cvtColor(color_segmented_map, color_segmented_map, CV_GRAY2BGR);
        for(size_t i = 1; i <= room_centers_x_values.size(); ++i)
        {
            //choose random color for each room
            const cv::Vec3b color((rand() % 250) + 1, (rand() % 250) + 1, (rand() % 250) + 1);
            for(size_t v = 0; v < indexed_map.rows; ++v)
                for(size_t u = 0; u < indexed_map.cols; ++u)
                    if(indexed_map.at<int>(v,u) == i)
                        color_segmented_map.at<cv::Vec3b>(v,u) = color;
        }
//		cv::Mat disp = segmented_map.clone();
        for (size_t index = 0; index < room_centers_x_values.size(); ++index)
            cv::circle(color_segmented_map, cv::Point(room_centers_x_values[index], room_centers_y_values[index]), 2, cv::Scalar(256), CV_FILLED);

        cv::imshow("segmentation", color_segmented_map);
        cv::waitKey();

        
    }

}
//cv::Mat gen_test_map()
//{
//    cv::Mat map(200,200,CV_8UC1);
//    for (int y = 50; y < 150; y++)
//    {
//        for (int x = 50; x < 150; x++)
//        {
//            map.at<unsigned char>(y, x) = 255;
//        }
//    }
//    for (int y = 50; y < 150; y++)
//    {
//        int  x=100;
//        map.at<unsigned char>(y, x) = 0;
//    }

////    for (int x = 50; x < 150; x++)
////    {
////        int y=100;
////        map.at<unsigned char>(y, x) = 0;
////    }
//    //!门1
//    for (int y = 95; y < 105; y++)
//    {
//        int  x=100;
//        map.at<unsigned char>(y, x) = 255;
//    }

//    return map;
//}
cv::Mat gen_test_map()
{
    cv::Mat map(400,400,CV_8UC1);
    for (int y = 100; y < 300; y++)
    {
        for (int x = 100; x < 300; x++)
        {
            map.at<unsigned char>(y, x) = 255;
        }
    }
    for (int y = 100; y < 300; y++)
    {
        int  x=200;
        map.at<unsigned char>(y, x) = 0;
    }


    //!门1
    for (int y = 190; y < 210; y++)
    {
        int  x=200;
        map.at<unsigned char>(y, x) = 255;
    }

    return map;
}

cv::Mat load_carto_map_by_file(cv::Mat& outOriMap)
{
    int row=1000;
    int col=1000;

    cv::Mat room_map(cvSize(row,col),CV_8UC1);

    cv::Mat oriMap(cvSize(row,col),CV_8UC1);

    int area_px = 0;
    FILE *fp=NULL;
    int i=0,j=0,value=0;
    fp = fopen("../map1000.txt","r");
    if(NULL==fp)
    {
        printf("load fail\n");
        return room_map;
    }
    while (feof(fp) == 0)
    {
        for(i=0;i<row;i++)
        {
            for(j=0;j<col;j++)
            {
                fscanf(fp,"%d\n",&value);
                if(UNKNOWN==value||BAR==value)
                {
                    room_map.at<uchar>(i,j)=0;
                }else{
                    room_map.at<uchar>(i,j)=255;
                    area_px++;
                }
                oriMap.at<uchar>(i,j)=value;
            }
        }

    }
    outOriMap=oriMap;
    return room_map;

}
void show_ori_nav_map(cv::Mat navMap)
{
     cv::Mat laberMap=navMap.clone();
    for(int i=0;i<navMap.rows;i++)
    {
        for(int j=0;j<navMap.cols;j++)
        {
            uchar st=navMap.at<uchar>(i,j);

            if(UNKNOWN==st)
            {
                laberMap.at<uchar>(i,j)=255;
            }else if(BAR==st)
            {
                 laberMap.at<uchar>(i,j)=0;
            }else
            {
                 laberMap.at<uchar>(i,j)=200;
            }

        }
    }
    cv::imshow("oriShowMap", laberMap);
    cv::waitKey();
}
cv::Mat convert_to_app_show_map(cv::Mat oriMap)
{
    cv::Mat laberMap=oriMap.clone();
    for(int i=0;i<oriMap.rows;i++)
    {
        for(int j=0;j<oriMap.cols;j++)
        {
            uchar st=oriMap.at<uchar>(i,j);
            //!黑色为0
            if(0==st)
            {
              laberMap.at<uchar>(i,j)=255;
            }else if(127==st)
            {
                 laberMap.at<uchar>(i,j)=0;
            }else if(255==st)
            {
                 laberMap.at<uchar>(i,j)=200;
            }

        }
    }

    return laberMap;
}
cv::Mat laber_unkown_in_grey(cv::Mat oriMap,cv::Mat navMap)
{
    cv::Mat laberMap=oriMap.clone();
    int dir[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};
    for(int i=1;i<oriMap.rows-1;i++)
    {
        for(int j=1;j<oriMap.cols-1;j++)
        {
            uchar st=oriMap.at<uchar>(i,j);
            //!黑色为0
            if(0==st)
            {
                bool bBorder=false;
                //!只要有一个是白色，就是边界
                for(int k=0;k<8;k++)
                {
                    if(255==oriMap.at<uchar>(i+dir[k][0],j+dir[k][1]))
                    {

                        bBorder=true;
                        //break;

                    }
                }
                if(bBorder)
                {
                    //!过滤掉未知和empty的交界
                    //!只要周围有一个是障碍物，就是边界
                    bool bNearHasBar=false;
                    for(int m=-1;m<=1;m++)
                    {
                        for(int n=-1;n<=1;n++)
                        {
                            if(3==navMap.at<uchar>(i+m,j+n))
                            {
                                bNearHasBar=true;
                                break;
                            }
                        }
                        if(bNearHasBar)
                        {
                            break;
                        }
                    }
                    if(false==bNearHasBar)
                    {
                        bBorder=false;
                    }

                }
                if(bBorder)
                {
                     laberMap.at<uchar>(i,j)=127;
                }

            }
        }
    }
    return laberMap;
}

cv::Mat laber_edge_in_grey(cv::Mat oriMap,cv::Mat edgeMap)
{
    cv::Mat laberMap=oriMap.clone();

    for(int i=1;i<oriMap.rows-1;i++)
    {
        for(int j=1;j<oriMap.cols-1;j++)
        {
            uchar st=edgeMap.at<uchar>(i,j);
            //!黑色为0
            if(255==st)
            {
               laberMap.at<uchar>(i,j)=127;
            }

        }
    }
    return laberMap;
}
cv::Mat detect_edge_map(cv::Mat oriMap,cv::Mat navMap)
{
    cv::Mat laberMap=cv::Mat::zeros(oriMap.rows,oriMap.cols,CV_8UC1);
    int dir[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{-1,1},{1,-1},{-1,-1}};
    for(int i=1;i<oriMap.rows-1;i++)
    {
        for(int j=1;j<oriMap.cols-1;j++)
        {
            uchar st=oriMap.at<uchar>(i,j);
            //!黑色为0
            if(0==st)
            {
                bool bBorder=false;
                //!只要有一个是白色，就是边界
                for(int k=0;k<8;k++)
                {
                    if(255==oriMap.at<uchar>(i+dir[k][0],j+dir[k][1]))
                    {

                        bBorder=true;
                        //break;

                    }
                }
                if(bBorder)
                {
                    //!过滤掉未知和empty的交界
                    //!只要周围有一个是障碍物，就是边界
                    bool bNearHasBar=false;
                    for(int m=-1;m<=1;m++)
                    {
                        for(int n=-1;n<=1;n++)
                        {
                            if(3==navMap.at<uchar>(i+m,j+n))
                            {
                                bNearHasBar=true;
                                break;
                            }
                        }
                        if(bNearHasBar)
                        {
                            break;
                        }
                    }
                    if(false==bNearHasBar)
                    {
                        bBorder=false;
                    }

                }
                if(bBorder)
                {
                     laberMap.at<uchar>(i,j)=255;
                }

            }
        }
    }
    return laberMap;
}
cv::Mat opt_map_show(cv::Mat oriMap,cv::Mat navMap)
{
   // cv::Mat kernel1 = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::Mat optMap;
    //!1.先膨胀3，再腐蚀3，就是开操作，使图像更加平滑
    cv::erode(oriMap,optMap,getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::dilate(optMap,optMap,getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::imshow("optMap", optMap);
    cv::waitKey();

    //optMap=oriMap;

//    cv::Mat edges;
//    cv::Canny(optMap, edges,100, 200);
    cv::Mat laberMap=laber_unkown_in_grey(optMap,navMap);

    cv::Mat appMap=convert_to_app_show_map(laberMap);
    cv::imshow("appMap", appMap);
    cv::waitKey();
    return optMap;
}
cv::Mat opt_map_show2(cv::Mat oriMap,cv::Mat navMap)
{
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat optMap;
    //!1.先膨胀3，再腐蚀3，就是开操作，使图像更加平滑

   cv::Mat blurMap;
   cv::GaussianBlur(oriMap, blurMap, cv::Size(3, 3), 1.0, 1.0);
 //  cv::bilateralFilter(oriMap, blurMap,5, 250, 60);

   //    cv::medianBlur(oriMap,blurMap, 3);
    cv::imshow("openOptMap", blurMap);
    cv::waitKey();
  //  cv::threshold(blurMap, optMap, 0, 255, cv::THRESH_OTSU|cv::THRESH_BINARY);
    cv::threshold(blurMap, optMap, 200, 255, cv::THRESH_BINARY);

//    cv::erode(optMap,optMap,kernel);
//    cv::dilate(optMap,optMap,kernel);

   // cv::filter2D(optMap, optMap, -1, kernel);
    //optMap=oriMap;
    cv::imshow("optMap", optMap);
    cv::waitKey();
//    cv::Mat edges;
//    cv::Canny(optMap, edges,100, 200);
    cv::Mat laberMap=laber_unkown_in_grey(optMap,navMap);

    cv::Mat appMap=convert_to_app_show_map(laberMap);
    cv::imshow("appMap", appMap);
    cv::waitKey();

    return optMap;
}

cv::Mat opt_map_show3(cv::Mat oriMap,cv::Mat navMap)
{
    //cv::Mat kernel1 = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::Mat optMap;
    cv::Mat blurMap;
    cv::GaussianBlur(oriMap, blurMap, cv::Size(3, 3), 1.0, 1.0);

    cv::imshow("openOptMap", blurMap);
    cv::waitKey();

    cv::threshold(blurMap, optMap, 200, 255, cv::THRESH_BINARY);

    cv::Mat edgeMap=detect_edge_map(optMap,navMap);
    cv::imshow("optMap", edgeMap);
    cv::waitKey();

    merge_lines(edgeMap.data, edgeMap.cols, edgeMap.rows, 2, 2, 30);
    cv::imshow("optedgeMap", edgeMap);
    cv::waitKey();

    cv::Mat laberMap=laber_edge_in_grey(optMap,edgeMap);
    cv::imshow("laberMap", laberMap);
    cv::waitKey();
    cv::Mat appMap=convert_to_app_show_map(laberMap);
    cv::imshow("appMap", appMap);
    cv::waitKey();




    return laberMap;
}
int main(int argc, char** argv)
{    
//   std::string fileName="../test_maps/lab_ipa.png";
//    cv::Mat map = cv::imread(fileName, 0);
//    printf("map.row=%d col=%d\n",map.rows,map.cols);
//    if(map.empty() )
//    {
//        printf("open map failed,ret f=%s\n",fileName.c_str());
//        return 0;
//    }
//    for (int y = 0; y < map.rows; y++)
//    {
//        for (int x = 0; x < map.cols; x++)
//        {
//            //find not reachable regions and make them black
//            if (map.at<unsigned char>(y, x) < 250)
//            {
//                map.at<unsigned char>(y, x) = 0;
//            }
//            //else make it white
//            else
//            {
//                map.at<unsigned char>(y, x) = 255;
//            }
//        }
//    }
//    cv::Mat map=gen_test_map();
    cv::Mat navMap;
    cv::Mat map=load_carto_map_by_file(navMap);

    show_ori_nav_map(navMap);
//     cv::imshow("map_ori", map);
//     cv::waitKey();
//     cv::Mat expMap;
//     cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
//     cv::morphologyEx(map, expMap, cv::MORPH_OPEN, kernel);
    cv::Mat appmap=opt_map_show3(map,navMap);
     // cv::imshow("map", edgeMap);
     // cv::waitKey();
    cv::Mat original_img = appmap;
    VoronoiSegmentation voronoi_segmentation; //voronoi segmentation method
    cv::Mat segmented_map;
    const double map_resolution=0.05f;
    // means the max/min area that an area separated by critical lines is allowed to have)
    const double room_upper_limit_voronoi_=120.0;
    const double room_lower_limit_voronoi_=1.53;
    //  #larger value sets a larger neighborhood for searching critical points --> int
    const int voronoi_neighborhood_index_=280;
    // #sets the maximal number of iterations to search for a neighborhood, also used for the vrf segmentation --> int
    const int max_iterations_=150;
   // #minimal distance factor between two critical points before one of it gets eliminated --> double
    const double min_critical_point_distance_factor_=0.5f;
    //#maximal area [m²] of a room that should be merged with its surrounding rooms,
    //also used for the voronoi random field segmentation
    const double max_area_for_merging_=12.5f;

    bool display_segmented_map_=true;
    bool DEBUG_DISPLAYS=true;
    Timer time;
    time.start();

    voronoi_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_voronoi_, room_upper_limit_voronoi_,
                                    voronoi_neighborhood_index_, max_iterations_,
                                    min_critical_point_distance_factor_,
                                    max_area_for_merging_, (display_segmented_map_&&DEBUG_DISPLAYS));
    time.stop();
    std::cout<<"seg take time"<<time.getElapsedTimeInSec()<<std::endl;

    show_segment_color_map(segmented_map);
    std::cout<<"hello world"<<std::endl;
    return 0;
}
