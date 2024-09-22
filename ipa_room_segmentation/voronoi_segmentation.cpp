#include <ipa_room_segmentation/voronoi_segmentation.h>

#include <ipa_room_segmentation/wavefront_region_growing.h>
#include <ipa_room_segmentation/contains.h>

#include <ipa_room_segmentation/timer.h>
#include <set>



VoronoiSegmentation::VoronoiSegmentation()
{

}


void VoronoiSegmentation::segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
		double room_area_factor_lower_limit, double room_area_factor_upper_limit, int neighborhood_index, int max_iterations,
		double min_critical_point_distance_factor, double max_area_for_merging, bool display_map)
{
	//****************Create the Generalized Voronoi-Diagram**********************
	//This function takes a given map and segments it with the generalized Voronoi-Diagram. It takes following steps:
	//	I. It calculates the generalized Voronoi-Diagram using the function createVoronoiGraph.
	//	II. It extracts the critical points, which show the border between two segments. This part takes these steps:
	//		1. Extract node-points of the Voronoi-Diagram, which have at least 3 neighbors.
	//		2. Reduce the leave-nodes (Point on graph with only one neighbor) of the graph until the reduction
	//		   hits a node-Point. This is done to reduce the lines along the real voronoi-graph, coming from the discretisation
	//		   of the contour.
	//		3. Find the critical points in the reduced graph by searching in a specified neighborhood for a local minimum
	//		   in distance to the nearest black pixel. The size of the epsilon-neighborhood is dynamic and goes larger
	//		   in small areas, so they are split into lesser regions.
	//	III. It gets the critical lines, which go from the critical point to its two nearest black pixels and separate the
	//		 regions from each other. This part does following steps:
	//			1. Get the discretized contours of the map and the holes, because these are the possible candidates for
	//			   basis-points.
	//			2. Find the basis-points for each critical-point by finding the two nearest neighbors of the vector from 1.
	//			   Also it saves the angle between the two vectors pointing from the critical-point to its two basis-points.
	//			3. Some critical-lines are too close to each other, so the next part eliminates some of them. For this the
	//			   algorithm checks, which critical points are too close to each other. Then it compares the angles of these
	//			   points, which were calculated in 3., and takes the one with the larger angle, because smaller angles
	//			   (like 90 degree) are more likely to be at edges of the map or are too close to the borders. If they have
	//			   the same angle, the point which comes first in the critical-point-vector is chosen (took good results for
	//			   me, but is only subjective).
	//			4. Draw the critical lines, selected by 3. in the map with color 0.
	//	IV. It finds the segments, which are seperated by the critical lines of III. and fills them with a random colour that
	//		hasn't been already used yet. For this it:
	//			1. It erodes the map with critical lines, so small gaps are closed, and finds the contours of the segments.
	//			   Only contours that are large/small enough are chosen to be drawn.
	//			2. It draws the contours from 1. in a map with a random colour. Contours that belong to holes are not drawn
	//			   into the map.
	//			3. Spread the colour-regions to the last white Pixels, using the watershed-region-spreading function.

	//*********************I. Calculate and draw the Voronoi-Diagram in the given map*****************
    Timer time;
    time.start();
	cv::Mat voronoi_map = map_to_be_labeled.clone();
    std::vector<cv::Point> v_voronoi_pts;
    FastCreateVoronoiGraph(voronoi_map,v_voronoi_pts); //voronoi-map for the segmentation-algorithm
	cv::imshow("voronoi_map1", voronoi_map);
	cv::waitKey();
   // createVoronoiGraph(voronoi_map);
    time.stop();
    std::cout<<"createVoronoiGraph take time"<<time.getElapsedTimeInSec()<<std::endl;
    time.start();


	//***************************II. extract the possible candidates for critical Points****************************
	// 1.extract the node-points that have at least three neighbors on the voronoi diagram
	//	node-points are points on the voronoi-graph that have at least 3 neighbors
	// 2.reduce the side-lines along the voronoi-graph by checking if it has only one neighbor until a node-point is reached
	//	--> make it white
	//	repeat a large enough number of times so the graph converges
	std::set<cv::Point, cv_Point_comp> node_points; //variable for node point extraction
    FastPruneVoronoiGraph(voronoi_map,v_voronoi_pts,node_points);
   // pruneVoronoiGraph(voronoi_map,node_points);
//    if(display_map == true)
//    {
//        cv::Mat display = voronoi_map.clone();
//        cv::imshow("voronoi_map", display);
//        cv::waitKey();
//    }
	cv::imshow("voronoi_map2", voronoi_map);
	cv::waitKey();
    time.stop();
    std::cout<<"pruneVoronoiGraph take time"<<time.getElapsedTimeInSec()<<std::endl;
    time.start();

	//3.find the critical points in the previously calculated generalized Voronoi-graph by searching in a specified
	//	neighborhood for the local minimum of distance to the nearest black pixel
	//	critical points need to have at least two neighbors (else they are end points, which would give a very small segment)

	//get the distance transformed map, which shows the distance of every white pixel to the closest zero-pixel
	cv::Mat distance_map; //distance-map of the original-map (used to check the distance of each point to nearest black pixel)
	cv::distanceTransform(map_to_be_labeled, distance_map, CV_DIST_L2, 5);
	cv::convertScaleAbs(distance_map, distance_map);


	std::vector<cv::Point> critical_points; //saving-variable for the critical points found on the Voronoi-graph
	for (int v = 0; v < voronoi_map.rows; v++)
	{
		for (int u = 0; u < voronoi_map.cols; u++)
		{
			if (voronoi_map.at<unsigned char>(v, u) == 127)
			{
				//make the size of the region to be checked dependent on the distance of the current pixel to the closest
				//zero-pixel, so larger areas are split into more regions and small areas into fewer
                int eps = neighborhood_index / (int) distance_map.at<unsigned char>(v, u); //310
               //int eps =120;
               // printf("eps=%d dis=%d\n",eps,(int) distance_map.at<unsigned char>(v, u));
				int loopcounter = 0; //if a part of the graph is not connected to the rest this variable helps to stop the loop
				std::vector<cv::Point> temporary_points;	//neighboring-variables, which are different for each point
				std::set<cv::Point, cv_Point_comp> neighbor_points;	//neighboring-variables, which are different for each point
				int neighbor_count = 0;		//variable to save the number of neighbors for each point
				neighbor_points.insert(cv::Point(u,v)); //add the current Point to the neighborhood
				//find every Point along the voronoi graph in a specified neighborhood
				do
				{
					loopcounter++;
					//check every point in the neighborhood for other neighbors connected to it
					for(std::set<cv::Point, cv_Point_comp>::iterator it_neighbor_points = neighbor_points.begin(); it_neighbor_points != neighbor_points.end(); it_neighbor_points++)
					{
						for (int row_counter = -1; row_counter <= 1; row_counter++)
						{
							for (int column_counter = -1; column_counter <= 1; column_counter++)
							{
								if (row_counter == 0 && column_counter == 0)
									continue;

								//check the neighboring points
								//(if it already is in the neighborhood it doesn't need to be checked again)
								const cv::Point& current_neighbor_point = *it_neighbor_points;
								const int nu = current_neighbor_point.x + column_counter;
								const int nv = current_neighbor_point.y + row_counter;
								if (nv >= 0 && nu >= 0 && nv < voronoi_map.rows && nu < voronoi_map.cols &&
									voronoi_map.at<unsigned char>(nv, nu) == 127 && neighbor_points.find(cv::Point(nu, nv))==neighbor_points.end())
								{
									neighbor_count++;
									temporary_points.push_back(cv::Point(nu, nv));
								}
							}
						}
					}
					//go trough every found point after all neighborhood points have been checked and add them to it
					for (int temporary_point_index = 0; temporary_point_index < temporary_points.size(); temporary_point_index++)
					{
						neighbor_points.insert(temporary_points[temporary_point_index]);
						//make the found points white in the voronoi-map (already looked at)
						voronoi_map.at<unsigned char>(temporary_points[temporary_point_index].y, temporary_points[temporary_point_index].x) = 255;
						voronoi_map.at<unsigned char>(v, u) = 255;
					}
					//check if enough neighbors have been checked or checked enough times (e.g. at a small segment of the graph)
				} while (neighbor_count <= eps && loopcounter < max_iterations);
				//check every found point in the neighborhood if it is the local minimum in the distanceMap
				cv::Point current_critical_point = cv::Point(u, v);
				for(std::set<cv::Point, cv_Point_comp>::iterator it_neighbor_points = neighbor_points.begin(); it_neighbor_points != neighbor_points.end(); it_neighbor_points++)
				{
					if (distance_map.at<unsigned char>(it_neighbor_points->y, it_neighbor_points->x) < distance_map.at<unsigned char>(current_critical_point.y, current_critical_point.x))
					{
						current_critical_point = cv::Point(*it_neighbor_points);
					}
				}
				//add the local minimum point to the critical points
				critical_points.push_back(current_critical_point);
			}
		}
	}

	cv::Mat test_map = map_to_be_labeled.clone();
	// 遍历所有关键点并在地图上绘制圆点
	for (size_t i = 0; i < critical_points.size(); ++i)
	{
		// 使用 cv::circle 函数绘制每个关键点
		// 参数1：目标图像
		// 参数2：关键点的坐标
		// 参数3：圆的半径
		// 参数4：颜色 (Scalar(0, 0, 255) 表示红色)
		// 参数5：圆的填充方式 (-1 表示填充整个圆)
		cv::circle(test_map, critical_points[i], 3, cv::Scalar(0, 0, 255), -1);
	}
	// 显示结果
	cv::imshow("Map with Critical Points", test_map);
	cv::waitKey(0);

	time.stop();
	std::cout << "get critical_points2 take time" << time.getElapsedTimeInSec() << std::endl;
	time.start();

	//    if(display_map == true)
	//    {
	//        cv::Mat display = map_to_be_labeled.clone();
	//        for (size_t i=0; i<critical_points.size(); ++i)
	//            cv::circle(display, critical_points[i], 2, cv::Scalar(128), -1);
	//        cv::imshow("critical points", display);
	//        cv::waitKey();
	//    }

	
	//
	//*************III. draw the critical lines from every found critical Point to its two closest zero-pixel****************
	//

	//map to draw the critical lines and fill the map with random colors
	map_to_be_labeled.convertTo(segmented_map, CV_32SC1, 256, 0); // rescale to 32 int, 255 --> 255*256 = 65280

	// 1. Get the points of the contour, which are the possible closest points for a critical point
	//clone the map to extract the contours, because after using OpenCV find-/drawContours
	//the map will be different from the original one
	cv::Mat temporary_map_to_extract_the_contours = segmented_map.clone();
	std::vector < std::vector<cv::Point> > contours;
	cv::findContours(temporary_map_to_extract_the_contours, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	
	// //test
	// std::vector<cv::Point> cornerPoints;
    // cv::goodFeaturesToTrack(map_to_be_labeled, cornerPoints, 100, 0.01, 10);
	// for (const auto& contour : contours) {
    //     if (contour.size() > 2) {
    //         cv::Point startPoint = contour.front();
    //         cv::Point endPoint = contour.back();
    //         cornerPoints.push_back(startPoint);
    //         cornerPoints.push_back(endPoint);
    //     }
    // }

	// std::vector<cv::Point> filtered_points;
	// for (const auto &point : cornerPoints)
	// {
	// 	int non_zero_count = 0;

	// 	// 遍历邻域
	// 	for (int dy = -1; dy <= 1; dy++)
	// 	{
	// 		for (int dx = -1; dx <= 1; dx++)
	// 		{
	// 			int nx = point.x + dx;
	// 			int ny = point.y + dy;

	// 			// 确保不越界
	// 			if (nx >= 0 && ny >= 0 && nx < voronoi_map.cols && ny < voronoi_map.rows)
	// 			{
	// 				// 检查是否为非零像素
	// 				if (voronoi_map.at<uchar>(ny, nx) != 0)
	// 				{
	// 					non_zero_count++;
	// 				}
	// 			}
	// 		}
	// 	}

	// 	// 如果邻域内的非零像素少于阈值，保留该点
	// 	if (non_zero_count == 1)
	// 	{
	// 		filtered_points.push_back(point);
	// 	}
	// }


	// for (const auto& point : filtered_points) {
    //     cv::circle(voronoi_map, point, 5, cv::Scalar(100, 0, 255), -1); // Green for corners
    // }



    // cv::imshow("filtered_points ", voronoi_map);
    // cv::waitKey(0);


	// 2. Get the basis-points for each critical-point
	std::vector<cv::Point> basis_points_1, basis_points_2;
	std::vector<double> length_of_critical_line;
	std::vector<double> angles; //the angles between the basis-lines of each critical Point
	for (int critical_point_index = 0; critical_point_index < critical_points.size(); critical_point_index++)
	{
		//set inital points and values for the basis points so the distance comparison can be done
		cv::Point basis_point_1 = contours[0][0];
		cv::Point basis_point_2 = contours[0][1];
		//inital values of the first vector from the current critical point to the contour points and for the distance of it
		const cv::Point& critical_point = critical_points[critical_point_index];
		double vector_x_1 = critical_point.x - contours[0][0].x;
		double vector_y_1 = critical_point.y - contours[0][0].y;
		double distance_basis_1 = std::sqrt(vector_x_1*vector_x_1 + vector_y_1*vector_y_1);
		//inital values of the second vector from the current critical point to the contour points and for the distance of it
		double vector_x_2 = critical_point.x - contours[0][1].x;
		double vector_y_2 = critical_point.y - contours[0][1].y;
		double distance_basis_2 = std::sqrt(vector_x_2*vector_x_2 + vector_y_2*vector_y_2);

		//find first basis point
		int basis_vector_1_x, basis_vector_2_x, basis_vector_1_y, basis_vector_2_y;
		for (int c = 0; c < contours.size(); c++)
		{
			for (int p = 0; p < contours[c].size(); p++)
			{
				//calculate the Euclidian distance from the critical Point to the Point on the contour
				const double vector_x = contours[c][p].x - critical_point.x;
				const double vector_y = contours[c][p].y - critical_point.y;
				const double current_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);
				//compare the distance to the saved distances if it is smaller
				if (current_distance < distance_basis_1)
				{
					distance_basis_1 = current_distance;
					basis_point_1 = contours[c][p];
					basis_vector_1_x = vector_x;
					basis_vector_1_y = vector_y;
				}
			}
		}
		//find second basisPpoint
		for (int c = 0; c < contours.size(); c++)
		{
			for (int p = 0; p < contours[c].size(); p++)
			{
				//calculate the Euclidian distance from the critical point to the point on the contour
				const double vector_x = contours[c][p].x - critical_point.x;
				const double vector_y = contours[c][p].y - critical_point.y;
				const double current_distance = std::sqrt(vector_x*vector_x + vector_y*vector_y);
				//calculate the distance between the current contour point and the first basis point to make sure they
				//are not too close to each other
				const double vector_x_basis = basis_point_1.x - contours[c][p].x;
				const double vector_y_basis = basis_point_1.y - contours[c][p].y;
				const double basis_distance = std::sqrt(vector_x_basis*vector_x_basis + vector_y_basis*vector_y_basis);
				if (current_distance > distance_basis_1 && current_distance < distance_basis_2 &&
					basis_distance > (double) distance_map.at<unsigned char>(critical_point.y, critical_point.x))
				{
					distance_basis_2 = current_distance;
					basis_point_2 = contours[c][p];
					basis_vector_2_x = vector_x;
					basis_vector_2_y = vector_y;
				}
			}
		}
		//calculate angle between the vectors from the critical Point to the found basis-points
		double current_angle = std::acos((basis_vector_1_x * basis_vector_2_x + basis_vector_1_y * basis_vector_2_y) / (distance_basis_1 * distance_basis_2)) * 180.0 / PI;

		//save the critical line with its calculated values
		basis_points_1.push_back(basis_point_1);
		basis_points_2.push_back(basis_point_2);
		length_of_critical_line.push_back(distance_basis_1 + distance_basis_2);
		angles.push_back(current_angle);
	}
   // std::vector<cv::Point> final_critical_points;
	//3. Check which critical points should be used for the segmentation. This is done by checking the points that are
	//   in a specified distance to each other and take the point with the largest calculated angle, because larger angles
	//   correspond to a separation across the room, which is more useful

// std::vector<cv::Point> contour_endpoints;
// std::vector<cv::Point> contour_corners;
// int certain_threshold = 10;
// double angle = 5;
// // 遍历每个轮廓
// for (int c = 0; c < contours.size(); c++) {
//     for (int p = 1; p < contours[c].size() - 1; p++) {
//         // 当前点及其相邻点
//         cv::Point prev_point = contours[c][p - 1];
//         cv::Point curr_point = contours[c][p];
//         cv::Point next_point = contours[c][p + 1];

//         // 计算相邻点之间的向量
//         cv::Point vec1 = prev_point - curr_point;
//         cv::Point vec2 = next_point - curr_point;

//         // 判断是否为端点
//         if (cv::norm(vec1) > certain_threshold) { // 该点与邻居的距离较大，认为是端点
//             contour_endpoints.push_back(curr_point);
//         }

//         // 判断是否为拐角点：两个向量夹角接近90度
//         double angle = std::acos((vec1.x * vec2.x + vec1.y * vec2.y) / (cv::norm(vec1) * cv::norm(vec2)));
//         if (std::abs(angle - CV_PI / 2) < angle) {
//             contour_corners.push_back(curr_point);
//         }
//     }
// }



	// 定义水平或垂直角度的阈值，比如5度
const double angle_threshold = 80;

// 遍历每一个关键点
for (int first_critical_point = 0; first_critical_point < critical_points.size(); first_critical_point++)
{
	bool draw = true; // 判断是否绘制该线
	for (int second_critical_point = 0; second_critical_point < critical_points.size(); second_critical_point++)
	{

		// // 获取两点的距离
        //     const double vector_x = critical_points[second_critical_point].x - critical_points[first_critical_point].x;
        //     const double vector_y = critical_points[second_critical_point].y - critical_points[first_critical_point].y;
        //     const double critical_point_distance = std::sqrt(vector_x * vector_x + vector_y * vector_y);

        //     // 判断这两个点是否是障碍物的端点或拐角点
        //     bool is_endpoint_1 = std::find(contour_endpoints.begin(), contour_endpoints.end(), critical_points[first_critical_point]) != contour_endpoints.end();
        //     bool is_corner_1 = std::find(contour_corners.begin(), contour_corners.end(), critical_points[first_critical_point]) != contour_corners.end();
        //     bool is_endpoint_2 = std::find(contour_endpoints.begin(), contour_endpoints.end(), critical_points[second_critical_point]) != contour_endpoints.end();
        //     bool is_corner_2 = std::find(contour_corners.begin(), contour_corners.end(), critical_points[second_critical_point]) != contour_corners.end();

        //     // 如果两个点都不是端点或拐角点，则不绘制
        //     if (!(is_endpoint_1 || is_corner_1) && !(is_endpoint_2 || is_corner_2)) {
        //         draw = false;
        //         break;
        //     }

        //     // 检查点的角度和距离条件
        //     if (critical_point_distance < ((int) distance_map.at<unsigned char>(critical_points[first_critical_point].y, critical_points[first_critical_point].x) * min_critical_point_distance_factor)) {
        //         if (angles[first_critical_point] < angles[second_critical_point]) {
        //             draw = false;
        //             break;
        //         }
        //         if (angles[first_critical_point] == angles[second_critical_point] &&
        //             length_of_critical_line[first_critical_point] > length_of_critical_line[second_critical_point] &&
        //             (length_of_critical_line[second_critical_point] > 3 || first_critical_point > second_critical_point)) {
        //             draw = false;
        //             break;
        //         }
        //     }



		if (second_critical_point != first_critical_point)
		{
			// 获取两个关键点之间的向量
			const double vector_x = critical_points[second_critical_point].x - critical_points[first_critical_point].x;
			const double vector_y = critical_points[second_critical_point].y - critical_points[first_critical_point].y;
			const double critical_point_distance = std::sqrt(vector_x * vector_x + vector_y * vector_y);

			// 检查两个关键点之间的距离是否过近
			if (critical_point_distance < ((int)distance_map.at<unsigned char>(critical_points[first_critical_point].y, critical_points[first_critical_point].x) * min_critical_point_distance_factor))
			{
				// 检查相邻关键点之间的角度
				if (angles[first_critical_point] < angles[second_critical_point])
				{
					draw = false;
					break;
				}
				// 如果角度相等，比较关键线的长度
				if (angles[first_critical_point] == angles[second_critical_point] &&
					length_of_critical_line[first_critical_point] > length_of_critical_line[second_critical_point] &&
					(length_of_critical_line[second_critical_point] > 3 || first_critical_point > second_critical_point))
				{
					draw = false;
					break;
				}
			}
		}
	}



	// 如果需要绘制关键线，则检查角度是否接近水平或垂直
	if (draw)
	{
		// 计算关键线的角度（使用之前计算的current_angle）
		double angle = angles[first_critical_point];

		// 检查是否接近水平或垂直（0度、90度、180度）
		if ((std::abs(angle) <= angle_threshold) ||		   // 接近水平线（0度）
			(std::abs(angle - 90.0) <= angle_threshold) || // 接近垂直线（90度）
			(std::abs(angle - 180.0) <= angle_threshold))  // 接近水平线（180度）
		{
			// 如果角度符合条件，则绘制关键线
			cv::line(voronoi_map, basis_points_1[first_critical_point], basis_points_2[first_critical_point], cv::Scalar(0), 2);
		}
		// 如果不满足水平或垂直条件，则不绘制
	}
}

	time.stop();
    std::cout<<"get critical_line take time"<<time.getElapsedTimeInSec()<<std::endl;
    time.start();
//    if(display_map == true)
//    {
//        cv::Mat display = map_to_be_labeled.clone();
//        for (size_t i=0; i<final_critical_points.size(); ++i)
//            cv::circle(display, final_critical_points[i], 2, cv::Scalar(128), -1);
//        cv::imshow("final_critical_points ", display);
//        cv::waitKey();
//    }
    //    if(display_map == true)
//    {
//        cv::imshow("voronoi_map", voronoi_map);
//        cv::waitKey();
//    }




	//***********************Find the Contours seperated from the critcal lines and fill them with color******************

	std::vector < cv::Scalar > already_used_colors; //saving-vector to save the already used coloures

	std::vector < cv::Vec4i > hierarchy; //variables for coloring the map

	std::vector<Room> rooms; //Vector to save the rooms in this map

    //1. Erode map one time, so small gaps are closed
//    cv::erode(voronoi_map_, voronoi_map_, cv::Mat(), cv::Point(-1, -1), 1);

	cv::findContours(voronoi_map, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);


    std::map<int, size_t> label_vector_index_codebook;
     size_t vector_index = 0;
	cv::imshow("voronoi_map3", voronoi_map);
	cv::waitKey();


	for (int current_contour = 0; current_contour < contours.size(); current_contour++)
	{ //only draw contours that aren't holes
		if (hierarchy[current_contour][3] == -1)
		{
			//calculate area for the contour and check if it is large enough to be a room
			double room_area = map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[current_contour]);
			if (room_area >= room_area_factor_lower_limit && room_area <= room_area_factor_upper_limit)
			{
				//2. Draw the region with a random color into the map if it is large/small enough
				bool drawn = false;
				int loop_counter = 0; //counter if the loop gets into a endless loop
				do
				{
					loop_counter++;
                    //0-32767
					int random_number = rand() % 52224 + 13056;
					cv::Scalar fill_colour(random_number);
					//check if color has already been used
					if (!contains(already_used_colors, fill_colour) || loop_counter > 1000)
					{
						cv::drawContours(segmented_map, contours, current_contour, fill_colour, 1);
						already_used_colors.push_back(fill_colour);
						Room current_room(random_number); //add the current Contour as a room
//						for (int point = 0; point < contours[current_contour].size(); point++) //add contour points to room
//						{
//							current_room.insertMemberPoint(cv::Point(contours[current_contour][point]), map_resolution_from_subscription);
//						}
						rooms.push_back(current_room);
                        label_vector_index_codebook[random_number] = vector_index;
                        vector_index++;
						drawn = true;
					}
				} while (!drawn);
			}
		}
	}
	std::cout << "Found " << rooms.size() << " rooms.\n";
//    if(display_map == true)
//    {
//        cv::imshow("segmented_map", segmented_map);
//        cv::waitKey();
//    }
    time.stop();
    std::cout<<"Find the Contours by critical_line take time"<<time.getElapsedTimeInSec()<<std::endl;
    time.start();
	//3.fill the last white areas with the surrounding color
    //wavefrontRegionGrowing(segmented_map);

    wavefrontRegionGrowingAndGetRec(segmented_map,label_vector_index_codebook,rooms);

    time.stop();
    std::cout<<"wavefrontRegionGrowing take time"<<time.getElapsedTimeInSec()<<std::endl;
    time.start();

//    if(display_map == true)
//    {
//        cv::imshow("before", segmented_map);
//        cv::waitKey(1);
//    }
//    display_map=false;
//	//4.merge the rooms together if neccessary

    display_map=false;
    //FastMergeRooms(segmented_map, rooms, map_resolution_from_subscription, max_area_for_merging, display_map);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));  // 使用3x3矩形结构元素
    time.stop();
    std::cout<<"mergeRooms take time"<<time.getElapsedTimeInSec()<<std::endl;


}
