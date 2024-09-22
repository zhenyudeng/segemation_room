#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ipa_room_segmentation/room_class.h>
void wavefrontRegionGrowing(cv::Mat& image);
void wavefrontRegionGrowingAndGetRec(cv::Mat& image, std::map<int, size_t> label_vector_index_codebook,
                                     std::vector<Room>& rooms);
