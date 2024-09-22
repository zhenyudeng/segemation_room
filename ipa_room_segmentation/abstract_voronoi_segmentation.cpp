#include <ipa_room_segmentation/voronoi_segmentation.h>

#include <ipa_room_segmentation/wavefront_region_growing.h>
#include <ipa_room_segmentation/contains.h>

#include <ipa_room_segmentation/timer.h>
#include <set>



AbstractVoronoiSegmentation::AbstractVoronoiSegmentation()
{

}

bool AbstractVoronoiSegmentation::determineRoomIndexFromRoomID(const std::vector<Room>& rooms, const int room_id, size_t& room_index)
{
	bool found_id = false;
	for (size_t r = 0; r < rooms.size(); r++)
	{
		if (rooms[r].getID() == room_id)
		{
			room_index = r;
			found_id = true;
			break;
		}
	}

	return found_id;
}

void AbstractVoronoiSegmentation::mergeRoomPair(std::vector<Room>& rooms, const int target_index, const int room_to_merge_index, cv::Mat& segmented_map, const double map_resolution)
{
	// integrate room to merge into target and delete merged room
	const int target_id = rooms[target_index].getID();
	const int room_to_merge_id = rooms[room_to_merge_index].getID();
	rooms[target_index].mergeRoom(rooms[room_to_merge_index], map_resolution);
	rooms[room_to_merge_index].setRoomId(rooms[target_index].getID(), segmented_map);
	rooms.erase(rooms.begin()+room_to_merge_index);
	std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);

	// update neighborhood statistics for remaining rooms
	for (size_t i=0; i<rooms.size(); ++i)
	{
		std::vector<int>& neighbor_ids = rooms[i].getNeighborIDs();
		std::vector<int>::iterator it = std::find(neighbor_ids.begin(), neighbor_ids.end(), room_to_merge_id);
		if (it != neighbor_ids.end())
		{
			std::vector<int>::iterator it2 = std::find(neighbor_ids.begin(), neighbor_ids.end(), target_id);
			if (it2 != neighbor_ids.end())
				neighbor_ids.erase(it);
			else
				*it = target_id;
		}

		std::map<int,int>& neighbor_statistics = rooms[i].getNeighborStatistics();
		std::map<int,int>::iterator it3 = neighbor_statistics.find(room_to_merge_id);
		if (it3 != neighbor_statistics.end())
		{
			std::map<int,int>::iterator it4 = neighbor_statistics.find(target_id);
			if (it4 != neighbor_statistics.end())
				it4->second += it3->second;
			else
				neighbor_statistics[target_id] = it3->second;
			neighbor_statistics.erase(it3);
		}
	}
}
void AbstractVoronoiSegmentation::FastMergeRoomPair(std::vector<Room>& rooms, const int target_index, const int room_to_merge_index, cv::Mat& segmented_map, const double map_resolution)
{
    // integrate room to merge into target and delete merged room
    const int target_id = rooms[target_index].getID();
    const int room_to_merge_id = rooms[room_to_merge_index].getID();
    rooms[target_index].FastMergeRoom(rooms[room_to_merge_index]);

    rooms[room_to_merge_index].FastSetRoomId(rooms[target_index].getID(), segmented_map);
   // rooms[room_to_merge_index].setRoomId(rooms[target_index].getID(), segmented_map);

    rooms.erase(rooms.begin()+room_to_merge_index);
    std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);

    // update neighborhood statistics for remaining rooms
    for (size_t i=0; i<rooms.size(); ++i)
    {
        std::vector<int>& neighbor_ids = rooms[i].getNeighborIDs();
        std::vector<int>::iterator it = std::find(neighbor_ids.begin(), neighbor_ids.end(), room_to_merge_id);
        if (it != neighbor_ids.end())
        {
            std::vector<int>::iterator it2 = std::find(neighbor_ids.begin(), neighbor_ids.end(), target_id);
            if (it2 != neighbor_ids.end())
                neighbor_ids.erase(it);
            else
                *it = target_id;
        }

        std::map<int,int>& neighbor_statistics = rooms[i].getNeighborStatistics();
        std::map<int,int>::iterator it3 = neighbor_statistics.find(room_to_merge_id);
        if (it3 != neighbor_statistics.end())
        {
            std::map<int,int>::iterator it4 = neighbor_statistics.find(target_id);
            if (it4 != neighbor_statistics.end())
                it4->second += it3->second;
            else
                neighbor_statistics[target_id] = it3->second;
            neighbor_statistics.erase(it3);
        }
    }
}
void AbstractVoronoiSegmentation::drawVoronoi(cv::Mat &img, const std::vector<std::vector<cv::Point2f> >& facets_of_voronoi, const cv::Scalar voronoi_color, const cv::Mat& eroded_map)
{
	// go trough each facet of the calculated Voronoi-graph and check if it should be drawn.
	for (std::vector<std::vector<cv::Point2f> >::const_iterator current_contour = facets_of_voronoi.begin(); current_contour != facets_of_voronoi.end(); ++current_contour)
	{
		// saving-variable for the last Point that has been looked at
		cv::Point2f last_point = current_contour->back();
		// draw each line of the voronoi-cell
		for (size_t c = 0; c < current_contour->size(); ++c)
		{
			// variable to check, whether a Point is inside a white area or not
			bool inside = true;
			cv::Point2f current_point = current_contour->at(c);
			// only draw lines that are inside the map-contour
			if (((int)current_point.x<0) || ((int)current_point.x >= eroded_map.cols) ||
				((int)current_point.y<0) || ((int)current_point.y >= eroded_map.rows) ||
				eroded_map.at<uchar>((int)current_point.y, (int)current_point.x) == 0 ||
				((int)last_point.x<0) || ((int)last_point.x >= eroded_map.cols) ||
				((int)last_point.y<0) || ((int)last_point.y >= eroded_map.rows) ||
				eroded_map.at<uchar>((int)last_point.y, (int)last_point.x) == 0)
				inside = false;
			if (inside)
			{
				cv::line(img, last_point, current_point, voronoi_color, 1);
			}
			last_point = current_point;
		}
	}
}

//****************Create the Generalized Voronoi-Diagram**********************
// This function is here to create the generalized voronoi-graph in the given map. It does following steps:
//	1. It finds every discretized contour in the given map (they are saved as vector<Point>). Then it takes these
//	   contour-Points and adds them to the OpenCV Delaunay generator from which the voronoi-cells can be generated.
//	2. Finally it gets the boundary-Points of the voronoi-cells with getVoronoiFacetList. It takes these facets
//	   and draws them using the drawVoronoi function. This function draws the facets that only have Points inside
//	   the map-contour (other lines go to not-reachable places and are not necessary to be looked at).
//	3. It returns the map that has the generalized voronoi-graph drawn in.
void AbstractVoronoiSegmentation::createVoronoiGraph(cv::Mat& map_for_voronoi_generation)
{
	cv::Mat map_to_draw_voronoi_in = map_for_voronoi_generation.clone(); //variable to save the given map for drawing in the voronoi-diagram

	cv::Mat temporary_map_to_calculate_voronoi = map_for_voronoi_generation.clone(); //variable to save the given map in the createVoronoiGraph-function

	//apply a closing-operator on the map so bad parts are neglected
    cv::erode(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi,cv::Mat());
    cv::dilate(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi,cv::Mat());

	//********************1. Get OpenCV delaunay-traingulation******************************
	cv::Rect rect(0, 0, map_to_draw_voronoi_in.cols, map_to_draw_voronoi_in.rows); //variables to generate the voronoi-diagram, using OpenCVs delaunay-triangulation
	cv::Subdiv2D subdiv(rect);
	std::vector<std::vector<cv::Point> > hole_contours; //variable to save the hole-contours (= black holes inside the white map)
	std::vector<std::vector<cv::Point> > contours; //variables for contour extraction and discretisation
	//hierarchy saves if the contours are hole-contours:
	//hierarchy[{0,1,2,3}]={next contour (same level), previous contour (same level), child contour, parent contour}
	//child-contour = 1 if it has one, = -1 if not, same for parent_contour
	std::vector<cv::Vec4i> hierarchy;

	//get contours of the map
	cv::Mat temp = map_to_draw_voronoi_in.clone();
	cv::findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	cv::drawContours(map_to_draw_voronoi_in, contours, -1, cv::Scalar(255), CV_FILLED);

//    if(1)
//    {
//        cv::Mat display = map_to_draw_voronoi_in.clone();
//        cv::imshow("map_to_draw_voronoi_in", display);
//        cv::waitKey();
//    }
   // int cCnt=0;
	//put every point of the map-contours into the Delaunay-generator of OpenCV
	for (int current_contour = 0; current_contour < contours.size(); current_contour++)
	{
		for (int current_point = 0; current_point < contours[current_contour].size(); current_point++)
		{
			cv::Point fp = contours[current_contour][current_point];
			subdiv.insert(fp);
           // printf("cCnt=%d x=%d y=%d\n",cCnt++,fp.x,fp.y);
		}
		//get the contours of the black holes --> it is necessary to check if points are inside these in drawVoronoi
		if (hierarchy[current_contour][2] == -1 && hierarchy[current_contour][3] != -1)
		{
			hole_contours.push_back(contours[current_contour]);
		}
	}

	//********************2. Get facets and draw voronoi-Graph******************************
	//erode the map so that points near the boundary are not drawn later (see drawVoronoi)
	cv::Mat eroded_map;
	cv::Point anchor(-1, -1);
	cv::erode(temporary_map_to_calculate_voronoi, eroded_map, cv::Mat(), anchor, 2);

	//get the Voronoi regions from the delaunay-subdivision graph
	const cv::Scalar voronoi_color(127); //define the voronoi-drawing colour
	std::vector<std::vector<cv::Point2f> > voronoi_facets; // variables to find the facets and centers of the voronoi-cells
	std::vector<cv::Point2f> voronoi_centers;
	subdiv.getVoronoiFacetList(std::vector<int>(), voronoi_facets, voronoi_centers);

	//draw the voronoi-regions into the map
	drawVoronoi(map_to_draw_voronoi_in, voronoi_facets, voronoi_color, eroded_map);

	//make pixels black, which were black before and were colored by the voronoi-regions
	for (int v = 0; v < map_to_draw_voronoi_in.rows; v++)
	{
		for (int u = 0; u < map_to_draw_voronoi_in.cols; u++)
		{
			if (map_for_voronoi_generation.at<unsigned char>(v, u) == 0)
			{
				map_to_draw_voronoi_in.at<unsigned char>(v, u) = 0;
			}
		}
	}
	map_for_voronoi_generation = map_to_draw_voronoi_in;
}
void AbstractVoronoiSegmentation::FastCreateVoronoiGraph(cv::Mat& map_for_voronoi_generation,std::vector<cv::Point>& v_voronoi_pts)
{
    cv::Mat map_to_draw_voronoi_in = map_for_voronoi_generation.clone(); //variable to save the given map for drawing in the voronoi-diagram

    cv::Mat temporary_map_to_calculate_voronoi = map_for_voronoi_generation.clone(); //variable to save the given map in the createVoronoiGraph-function

    //apply a closing-operator on the map so bad parts are neglected
    //!外面已经做了腐蚀和膨胀，所以这里可以不用了
    cv::erode(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi,cv::Mat());
    cv::dilate(temporary_map_to_calculate_voronoi, temporary_map_to_calculate_voronoi,cv::Mat());

    //********************1. Get OpenCV delaunay-traingulation******************************
    cv::Rect rect(0, 0, map_to_draw_voronoi_in.cols, map_to_draw_voronoi_in.rows); //variables to generate the voronoi-diagram, using OpenCVs delaunay-triangulation
    cv::Subdiv2D subdiv(rect);
    std::vector<std::vector<cv::Point> > hole_contours; //variable to save the hole-contours (= black holes inside the white map)
    std::vector<std::vector<cv::Point> > contours; //variables for contour extraction and discretisation
    //hierarchy saves if the contours are hole-contours:
    //hierarchy[{0,1,2,3}]={next contour (same level), previous contour (same level), child contour, parent contour}
    //child-contour = 1 if it has one, = -1 if not, same for parent_contour
    std::vector<cv::Vec4i> hierarchy;

    //get contours of the map
    cv::Mat temp = map_to_draw_voronoi_in.clone();
    cv::findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    cv::drawContours(map_to_draw_voronoi_in, contours, -1, cv::Scalar(255), CV_FILLED);

//    if(1)
//    {
//        cv::Mat display = map_to_draw_voronoi_in.clone();
//        cv::imshow("map_to_draw_voronoi_in", display);
//        cv::waitKey();
//    }
   // int cCnt=0;
    //put every point of the map-contours into the Delaunay-generator of OpenCV
    for (int current_contour = 0; current_contour < contours.size(); current_contour++)
    {
        for (int current_point = 0; current_point < contours[current_contour].size(); current_point++)
        {
            cv::Point fp = contours[current_contour][current_point];
            subdiv.insert(fp);
           // printf("cCnt=%d x=%d y=%d\n",cCnt++,fp.x,fp.y);
        }
        //get the contours of the black holes --> it is necessary to check if points are inside these in drawVoronoi
        if (hierarchy[current_contour][2] == -1 && hierarchy[current_contour][3] != -1)
        {
            hole_contours.push_back(contours[current_contour]);
        }
    }

    //********************2. Get facets and draw voronoi-Graph******************************
    //erode the map so that points near the boundary are not drawn later (see drawVoronoi)
    cv::Mat eroded_map;
    cv::Point anchor(-1, -1);
    cv::erode(temporary_map_to_calculate_voronoi, eroded_map, cv::Mat(), anchor, 2);

    //get the Voronoi regions from the delaunay-subdivision graph
    const cv::Scalar voronoi_color(127); //define the voronoi-drawing colour
    std::vector<std::vector<cv::Point2f> > voronoi_facets; // variables to find the facets and centers of the voronoi-cells
    std::vector<cv::Point2f> voronoi_centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), voronoi_facets, voronoi_centers);

    //draw the voronoi-regions into the map
    drawVoronoi(map_to_draw_voronoi_in, voronoi_facets, voronoi_color, eroded_map);

    //make pixels black, which were black before and were colored by the voronoi-regions
    for (int v = 0; v < map_to_draw_voronoi_in.rows; v++)
    {
        for (int u = 0; u < map_to_draw_voronoi_in.cols; u++)
        {
            if (map_for_voronoi_generation.at<unsigned char>(v, u) == 0)
            {
                map_to_draw_voronoi_in.at<unsigned char>(v, u) = 0;
            }
            if(127==map_to_draw_voronoi_in.at<unsigned char>(v, u))
            {
                //!x对应列对应u
                //!y对应行对应v
                cv::Point pt(u,v);
                v_voronoi_pts.push_back(pt);
            }
        }
    }
    map_for_voronoi_generation = map_to_draw_voronoi_in;
}
void AbstractVoronoiSegmentation::FastPruneVoronoiGraph(cv::Mat& voronoi_map,
                                                        const std::vector<cv::Point>& v_voronoi_pts,
                                                        std::set<cv::Point, cv_Point_comp>& node_points)
{
    // 1.extract the node-points that have at least three neighbors on the voronoi diagram
    //	node-points are points on the voronoi-graph that have at least 3 neighbors
    for(size_t i=0;i<v_voronoi_pts.size();i++)
    {
        //v代表行，存在y中
        int v=v_voronoi_pts[i].y;
        int u=v_voronoi_pts[i].x;
        int neighbor_count = 0;	// variable to save the number of neighbors for each point
        // check 3x3 region around current pixel
        for (int row_counter = -1; row_counter <= 1; row_counter++)
        {
            for (int column_counter = -1; column_counter <= 1; column_counter++)
            {
                // don't check the point itself
                if (row_counter == 0 && column_counter == 0)
                    continue;

                //check if neighbors are colored with the voronoi-color
                if (voronoi_map.at<unsigned char>(v + row_counter, u + column_counter) == 127)
                {
                    neighbor_count++;
                }
            }
        }
        if (neighbor_count > 2)
        {
            node_points.insert(cv::Point(u,v));
        }

    }
    // 2.reduce the side-lines along the voronoi-graph by checking if it has only one neighbor until a node-point is reached
    //	--> make it white
    //	repeat a large enough number of times so the graph converges
    std::vector<bool> v_b_delete(v_voronoi_pts.size(),false);
    for (int step = 0; step < 100; step++)
    {
        int cur_del_cnt=0;
        for(size_t i=0;i<v_voronoi_pts.size();i++)
        {
             if(false==v_b_delete[i])
             {
                 int v=v_voronoi_pts[i].y;
                 int u=v_voronoi_pts[i].x;
                 int neighbor_count = 0;		//variable to save the number of neighbors for each point
                 for (int row_counter = -1; row_counter <= 1; row_counter++)
                 {
                     for (int column_counter = -1; column_counter <= 1; column_counter++)
                     {
                         // don't check the point itself
                         if (row_counter == 0 && column_counter == 0)
                             continue;

                         // check the surrounding points
                         const int nv = v + row_counter;
                         const int nu = u + column_counter;
                         if (nv >= 0 && nu >= 0 && nv < voronoi_map.rows && nu < voronoi_map.cols && voronoi_map.at<unsigned char>(nv, nu) == 127)
                         {
                             neighbor_count++;
                         }
                     }
                 }
                 //if the current point is a node point found in the previous step, it belongs to the voronoi-graph
                 if (neighbor_count <= 1 && node_points.find(cv::Point(u,v)) == node_points.end())
                 {
                     //if the Point isn't on the voronoi-graph make it white
                     voronoi_map.at<unsigned char>(v, u) = 255;
                     v_b_delete[i]=true;
                     cur_del_cnt++;
                 }
             }
        }
        if(0==cur_del_cnt)
        {
            break;
        }
    }
}

void AbstractVoronoiSegmentation::pruneVoronoiGraph(cv::Mat& voronoi_map, std::set<cv::Point, cv_Point_comp>& node_points)
{
	// 1.extract the node-points that have at least three neighbors on the voronoi diagram
	//	node-points are points on the voronoi-graph that have at least 3 neighbors
	for (int v = 1; v < voronoi_map.rows-1; v++)
	{
		for (int u = 1; u < voronoi_map.cols-1; u++)
		{
			if (voronoi_map.at<unsigned char>(v, u) == 127)
			{
				int neighbor_count = 0;	// variable to save the number of neighbors for each point
				// check 3x3 region around current pixel
				for (int row_counter = -1; row_counter <= 1; row_counter++)
				{
					for (int column_counter = -1; column_counter <= 1; column_counter++)
					{
						// don't check the point itself
						if (row_counter == 0 && column_counter == 0)
							continue;

						//check if neighbors are colored with the voronoi-color
						if (voronoi_map.at<unsigned char>(v + row_counter, u + column_counter) == 127)
						{
							neighbor_count++;
						}
					}
				}
				if (neighbor_count > 2)
				{
					node_points.insert(cv::Point(u,v));
				}
			}
		}
	}

	// 2.reduce the side-lines along the voronoi-graph by checking if it has only one neighbor until a node-point is reached
	//	--> make it white
	//	repeat a large enough number of times so the graph converges
    for (int step = 0; step < 100; step++)
	{
        int notOkCnt=0;
		for (int v = 0; v < voronoi_map.rows; v++)
		{
			for (int u = 0; u < voronoi_map.cols; u++)
			{
				// set that the point is a point along the graph and not a side-line
				if (voronoi_map.at<unsigned char>(v, u) == 127)
				{
					int neighbor_count = 0;		//variable to save the number of neighbors for each point
					for (int row_counter = -1; row_counter <= 1; row_counter++)
					{
						for (int column_counter = -1; column_counter <= 1; column_counter++)
						{
							// don't check the point itself
							if (row_counter == 0 && column_counter == 0)
								continue;

							// check the surrounding points
							const int nv = v + row_counter;
							const int nu = u + column_counter;
							if (nv >= 0 && nu >= 0 && nv < voronoi_map.rows && nu < voronoi_map.cols && voronoi_map.at<unsigned char>(nv, nu) == 127)
							{
								neighbor_count++;
							}
						}
					}
					//if the current point is a node point found in the previous step, it belongs to the voronoi-graph
                    if (neighbor_count <= 1 && node_points.find(cv::Point(u,v)) == node_points.end())
                    {
						//if the Point isn't on the voronoi-graph make it white
						voronoi_map.at<unsigned char>(v, u) = 255;
                        notOkCnt++;
					}
				}
			}
		}
        if(0==notOkCnt)
        {
            break;
        }
        printf("step=%d notOkCnt=%d\n",step,notOkCnt);
	}
}

void AbstractVoronoiSegmentation::mergeRooms(cv::Mat& map_to_merge_rooms, std::vector<Room>& rooms, double map_resolution_from_subscription, double max_area_for_merging, bool display_map)
{
	// This function takes the segmented Map from the original Voronoi-segmentation-algorithm and merges rooms together,
	// that are small enough and have only two or one neighbor.

	// 1. go trough every pixel and add points to the rooms with the same ID
	for (int y = 0; y < map_to_merge_rooms.rows; y++)
	{
		for (int x = 0; x < map_to_merge_rooms.cols; x++)
		{
			int current_id = map_to_merge_rooms.at<int>(y, x);
			if (current_id != 0)
			{
				for (size_t current_room = 0; current_room < rooms.size(); current_room++) //add the Points with the same Id as a room to it
				{
					if (rooms[current_room].getID() == current_id) //insert the current point into the corresponding room
					{
						rooms[current_room].insertMemberPoint(cv::Point(x, y), map_resolution_from_subscription);
						break;
					}
				}
			}
		}
	}

	// 2. add the neighbor IDs for every point
	for (int current_room = 0; current_room < rooms.size(); current_room++)
	{
		const int current_id = rooms[current_room].getID();
		std::vector<cv::Point> considered_neighbors;		// storage for already counted neighborhood points
		const std::vector<cv::Point>& current_points = rooms[current_room].getMembers();
		for (int current_point = 0; current_point < current_points.size(); current_point++)
		{
			for (int row_counter = -1; row_counter <= 1; row_counter++)
			{
				for (int col_counter = -1; col_counter <= 1; col_counter++)
				{
					const int label = map_to_merge_rooms.at<int>(current_points[current_point].y + row_counter, current_points[current_point].x + col_counter);

					// collect neighbor IDs
					if (label != 0 && label != current_id)
						rooms[current_room].addNeighborID(label);

					// neighborhood statistics
					cv::Point neighbor_point(current_points[current_point].x + col_counter, current_points[current_point].y + row_counter);
					if (!contains(considered_neighbors, neighbor_point) && label != current_id)
					{
						rooms[current_room].addNeighbor(label);
						considered_neighbors.push_back(neighbor_point);
					}
				}
			}
		}
	}

	// 3. merge criteria
	// sort rooms ascending by area
	std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);
	// a) rooms with one neighbor and max. 75% walls around
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		if (current_room.getNeighborCount() == 1 && current_room.getArea() < max_area_for_merging && current_room.getWallToPerimeterRatio() <= 0.75)
		{
			// check every room if it should be merged with its neighbor that it shares the most boundary with
			merge_rooms = determineRoomIndexFromRoomID(rooms, current_room.getNeighborWithLargestCommonBorder(), merge_index);
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("a", map_to_merge_rooms);

	// b) small rooms
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if (current_room.getArea() < 2.0 && (double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.2)
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
			if ((double)rooms[merge_index].getWallToPerimeterRatio() > 0.8) //0.8
				merge_rooms = false;
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("b", map_to_merge_rooms);

	// c) merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		// merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if ((current_room.getNeighborCount()==1 || current_room.getPerimeterRatioOfXLargestRooms(1)>0.98) && current_room.getWallToPerimeterRatio() > 0.5 &&
			(double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.15)
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
			if (rooms[merge_index].getNeighborCount() > 2 && rooms[merge_index].getPerimeterRatioOfXLargestRooms(2)<0.95) // || rooms[merge_index].getWallToPerimeterRatio() < 0.4)
				merge_rooms = false;
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("c", map_to_merge_rooms);

	// d) merge rooms that share a significant part of their perimeter
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		std::map< int,int,std::greater<int> > neighbor_room_statistics_inverse;	// common border length, room_id
		current_room.getNeighborStatisticsInverse(neighbor_room_statistics_inverse);
//		std::vector<int>& neighbor_ids = current_room.getNeighborIDs();
//		for (size_t n=0; n<neighbor_ids.size(); ++n)
		for (std::map< int,int,std::greater<int> >::iterator it=neighbor_room_statistics_inverse.begin(); it!=neighbor_room_statistics_inverse.end(); ++it)
		{
			if (it->second==0)
				continue;		// skip wall

			const double neighbor_border_ratio = (double)current_room.getNeighborStatistics()[it->second]/current_room.getPerimeter();
			if (neighbor_border_ratio > 0.2 || (neighbor_border_ratio > 0.1 && current_room.getWallToPerimeterRatio() > (1-2*neighbor_border_ratio-0.05) && current_room.getWallToPerimeterRatio() < (1-neighbor_border_ratio)))
			{
				// merge with that neighbor that shares the most neighboring pixels
				merge_rooms = determineRoomIndexFromRoomID(rooms, it->second, merge_index);
				if ((double)rooms[merge_index].getNeighborStatistics()[current_room.getID()]/rooms[merge_index].getPerimeter() <= 0.1)
					merge_rooms = false;
				if (merge_rooms == true)
					break;
			}
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
	if (display_map == true)
		cv::imshow("d", map_to_merge_rooms);

	// e) largest room neighbor touches > 0.5 perimeter (happens often with furniture)
	for (int current_room_index = 0; current_room_index < rooms.size(); )
	{
		Room& current_room = rooms[current_room_index];
		bool merge_rooms = false;
		size_t merge_index = 0;

		const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
		if ((double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.4)
		{
			// merge with that neighbor that shares the most neighboring pixels
			merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
		}

		if (merge_rooms == true)
		{
			//std::cout << "merge " << current_room.getCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].getCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
			mergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
			current_room_index = 0;
		}
		else
			current_room_index++;
	}
}



void AlignRoomBoundariesToRectangles(cv::Mat& map_to_merge_rooms, std::vector<Room>& rooms) {
    for (int current_room = 0; current_room < rooms.size(); current_room++) {
        Room& room = rooms[current_room];
        const std::set<cv::Point, cv_Point_comp>& edge_points = room.getEdgePoint();  // 获取房间的边界点
        std::set<cv::Point, cv_Point_comp> new_edge_points;  // 存储调整后的边界点

        for (const cv::Point& point : edge_points) {
            int x = point.x;
            int y = point.y;

            // 查看周围8个方向上的点
            int neighbor_x[] = {-1, 0, 1, -1, 1, -1, 0, 1};
            int neighbor_y[] = {-1, -1, -1, 0, 0, 1, 1, 1};

            double best_angle = 90.0;  // 最好的角度初始为90度，即垂直或水平
            cv::Point best_point = point;

            for (int i = 0; i < 8; i++) {
                int nx = x + neighbor_x[i];
                int ny = y + neighbor_y[i];

                if (nx >= 0 && nx < map_to_merge_rooms.cols && ny >= 0 && ny < map_to_merge_rooms.rows) {
                    if (map_to_merge_rooms.at<int>(ny, nx) == room.getID()) {
                        // 计算与当前边界点的连线角度
                        double angle = atan2(ny - y, nx - x) * 180.0 / CV_PI;

                        // 如果角度接近水平（0或180度）或垂直（90度）
                        if (fabs(angle) < best_angle || fabs(angle - 90) < best_angle) {
                            best_angle = fabs(angle);  // 更新最佳角度
                            best_point = cv::Point(nx, ny);
                        }
                    }
                }
            }

            // 根据最佳角度决定是否将点对齐到水平或垂直线
            if (fabs(best_angle) < 45) {
                // 对齐到水平线
                new_edge_points.insert(cv::Point(x, best_point.y));
            } else {
                // 对齐到垂直线
                new_edge_points.insert(cv::Point(best_point.x, y));
            }
        }

        // 将新的边界点插入房间

        for (const cv::Point& p : new_edge_points) {
            room.insertEdgePoint(p);
        }

        // 更新 map_to_merge_rooms 中的边界信息
        for (const cv::Point& p : new_edge_points) {
            map_to_merge_rooms.at<int>(p.y, p.x) = room.getID();  // 将新的边界点写回地图
        }
    }

    // 显示更新后的地图
    cv::imshow("Updated Map", map_to_merge_rooms);
    cv::waitKey();
}






void AbstractVoronoiSegmentation::FastMergeRooms(cv::Mat& map_to_merge_rooms, std::vector<Room>& rooms, double map_resolution_from_subscription, double max_area_for_merging, bool display_map)
{
    // This function takes the segmented Map from the original Voronoi-segmentation-algorithm and merges rooms together,
    // that are small enough and have only two or one neighbor.

    // 1. go trough every pixel and add points to the rooms with the same ID
    //*********************I. Calculate and draw the Voronoi-Diagram in the given map*****************
   // display_map=true;
    Timer time;
    time.start();
	


	for (int current_room = 0; current_room < rooms.size(); current_room++)
	{
		room_rec_st rec=rooms[current_room].getRoomRec();
		int room_id=rooms[current_room].getID();
		int room_area_size_by_x=0;
		//!先x方向迭代
		for (int y = rec.min_y; y <= rec.max_y; y++)
		{
			int tmp_break_min_x=rec.min_x;
			for (int x = rec.min_x; x <= rec.max_x; x++)
			{
				if(room_id==map_to_merge_rooms.at<int>(y, x))
				{
					tmp_break_min_x=x;
					rooms[current_room].insertEdgePoint(cv::Point(x, y));
					break;
				}
			}
			int tmp_break_max_x=rec.max_x;
			for (int x = rec.max_x; x >=tmp_break_min_x ; x--)
			{
				if(room_id==map_to_merge_rooms.at<int>(y, x))
				{
					tmp_break_max_x=x;
					rooms[current_room].insertEdgePoint(cv::Point(x, y));
					break;
				}
			}
			//!粗略统计区域面积
			int diff_x=tmp_break_max_x-tmp_break_min_x-1;
			if(diff_x>0)
			{
				room_area_size_by_x+=diff_x;
			}
		}
		int room_area_size_by_y=0;
        //!再y方向迭代
        for (int x = rec.min_x; x <= rec.max_x; x++)

        {
            int tmp_break_min_y=rec.min_y;

             for (int y = rec.min_y; y <= rec.max_y; y++)
             {
                if(room_id==map_to_merge_rooms.at<int>(y, x))
                {
                    tmp_break_min_y=y;
                    rooms[current_room].insertEdgePoint(cv::Point(x, y));
                    break;
                }
            }
            int tmp_break_max_y=rec.max_y;
            for (int y = rec.max_y; y>=tmp_break_min_y; y--)
            {
                if(room_id==map_to_merge_rooms.at<int>(y, x))
                {
                    tmp_break_max_y=y;
                    rooms[current_room].insertEdgePoint(cv::Point(x, y));
                    break;
                }
            }
            //!粗略统计区域面积
            int diff_y=tmp_break_max_y-tmp_break_min_y-1;
            if(diff_y>0)
            {
                room_area_size_by_y+=diff_y;
            }


        }
        //!设置面积大小
        int min_area_size=0;
        if(room_area_size_by_x<room_area_size_by_y)
        {
            min_area_size=room_area_size_by_x;
        }else{
            min_area_size=room_area_size_by_y;
        }
        double area_size=map_resolution_from_subscription * map_resolution_from_subscription * min_area_size;

       // printf("set min_area_size=%d area_size=%lf\n",min_area_size,area_size);
        rooms[current_room].setArea(area_size);
    }
    //cv::Mat showMap=cv::Mat::zeros(map_to_merge_rooms.rows,map_to_merge_rooms.cols,CV_8UC1);
    // 2. add the neighbor IDs for every point
    for (int current_room = 0; current_room < rooms.size(); current_room++)
    {
        const int current_id = rooms[current_room].getID();
        std::set<cv::Point, cv_Point_comp> considered_neighbors;		// storage for already counted neighborhood points
        const std::set<cv::Point, cv_Point_comp>& current_points = rooms[current_room].getEdgePoint();
        //printf("current_points.size=%d\n",(int)current_points.size());
        for(std::set<cv::Point, cv_Point_comp>::iterator it= current_points.begin(); it != current_points.end(); it++)
        {
             const cv::Point& cur_point = *it;
      //      showMap.at<uchar>( cur_point.y , cur_point.x )=255;
            for (int row_counter = -1; row_counter <= 1; row_counter++)
            {
                for (int col_counter = -1; col_counter <= 1; col_counter++)
                {
                   // const cv::Point& cur_point = *it;
                    const int label = map_to_merge_rooms.at<int>(cur_point.y + row_counter, cur_point.x + col_counter);

                    // collect neighbor IDs
                    if (label != 0 && label != current_id){
                        rooms[current_room].addNeighborID(label);
                    }


                    // neighborhood statistics
                    cv::Point neighbor_point(cur_point.x + col_counter, cur_point.y + row_counter);
                    if (label != current_id&&considered_neighbors.end()==considered_neighbors.find(neighbor_point))
                    {
                        rooms[current_room].addNeighbor(label);
                       // showMap.at<uchar>( cur_point.y + row_counter, cur_point.x + col_counter)=255;
                        considered_neighbors.insert(neighbor_point);
                    }
                }
            }
        }
    }
//    cv::imshow("map_to_merge_rooms", map_to_merge_rooms);

//    cv::imshow("showMap", showMap);
//    cv::waitKey();
//    time.stop();

    std::cout<<"get edge take time"<<time.getElapsedTimeInSec()<<std::endl;
      time.start();
    // 3. merge criteria
    // sort rooms ascending by area
    std::sort(rooms.begin(), rooms.end(), sortRoomsAscending);
    // a) rooms with one neighbor and max. 75% walls around
    for (int current_room_index = 0; current_room_index < rooms.size(); )
    {
        Room& current_room = rooms[current_room_index];
        bool merge_rooms = false;
        size_t merge_index = 0;

        if (current_room.getNeighborCount() == 1 && current_room.getArea() < max_area_for_merging && current_room.getWallToPerimeterRatio() <= 0.75)
        {
            // check every room if it should be merged with its neighbor that it shares the most boundary with
            merge_rooms = determineRoomIndexFromRoomID(rooms, current_room.getNeighborWithLargestCommonBorder(), merge_index);
        }

        if (merge_rooms == true)
        {
            std::cout << "a merge " << current_room.FastGetCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].FastGetCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
            FastMergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
            current_room_index = 0;
        }
        else
            current_room_index++;
    }
//    if (display_map == true)
//    {
//        cv::imshow("a", map_to_merge_rooms);
//        cv::waitKey();
//    }


    // b) small rooms
    for (int current_room_index = 0; current_room_index < rooms.size(); )
    {
        Room& current_room = rooms[current_room_index];
        bool merge_rooms = false;
        size_t merge_index = 0;

        const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
        if (current_room.getArea() < 2.0 && (double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.2)
        {
            // merge with that neighbor that shares the most neighboring pixels
            merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
            if ((double)rooms[merge_index].getWallToPerimeterRatio() > 0.8) //0.8
                merge_rooms = false;
        }

        if (merge_rooms == true)
        {
            std::cout << "b merge " << current_room.FastGetCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].FastGetCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
            FastMergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
            current_room_index = 0;
        }
        else
            current_room_index++;
    }
//    if (display_map == true)
//    {
//        cv::imshow("b", map_to_merge_rooms);
//        cv::waitKey();
//    }

    // c) merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
    for (int current_room_index = 0; current_room_index < rooms.size(); )
    {
        Room& current_room = rooms[current_room_index];
        bool merge_rooms = false;
        size_t merge_index = 0;

        // merge a room with one neighbor that has max. 2 neighbors and sufficient wall ratio (connect parts inside a room)
        const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
        if ((current_room.getNeighborCount()==1 || current_room.getPerimeterRatioOfXLargestRooms(1)>0.98) && current_room.getWallToPerimeterRatio() > 0.5 &&
            (double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.15)
        {
            // merge with that neighbor that shares the most neighboring pixels
            merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
            if (rooms[merge_index].getNeighborCount() > 2 && rooms[merge_index].getPerimeterRatioOfXLargestRooms(2)<0.95) // || rooms[merge_index].getWallToPerimeterRatio() < 0.4)
                merge_rooms = false;
        }

        if (merge_rooms == true)
        {
            std::cout << "c merge " << current_room.FastGetCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].FastGetCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
            FastMergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
            current_room_index = 0;
        }
        else
            current_room_index++;
    }
//    if (display_map == true)
//    {
//        cv::imshow("c", map_to_merge_rooms);
//        cv::waitKey();
//    }

    // d) merge rooms that share a significant part of their perimeter
    for (int current_room_index = 0; current_room_index < rooms.size(); )
    {
        Room& current_room = rooms[current_room_index];
        bool merge_rooms = false;
        size_t merge_index = 0;

        std::map< int,int,std::greater<int> > neighbor_room_statistics_inverse;	// common border length, room_id
        current_room.getNeighborStatisticsInverse(neighbor_room_statistics_inverse);
//		std::vector<int>& neighbor_ids = current_room.getNeighborIDs();
//		for (size_t n=0; n<neighbor_ids.size(); ++n)
        for (std::map< int,int,std::greater<int> >::iterator it=neighbor_room_statistics_inverse.begin(); it!=neighbor_room_statistics_inverse.end(); ++it)
        {
            if (it->second==0)
                continue;		// skip wall

            const double neighbor_border_ratio = (double)current_room.getNeighborStatistics()[it->second]/current_room.getPerimeter();
            if (neighbor_border_ratio > 0.2 ||
                    (neighbor_border_ratio > 0.1 && current_room.getWallToPerimeterRatio() > (1-2*neighbor_border_ratio-0.05) && current_room.getWallToPerimeterRatio() < (1-neighbor_border_ratio)))
            {
                // merge with that neighbor that shares the most neighboring pixels
                merge_rooms = determineRoomIndexFromRoomID(rooms, it->second, merge_index);
                double neiSta=rooms[merge_index].getNeighborStatistics()[current_room.getID()];
                double per=rooms[merge_index].getPerimeter();
                double ratio=neiSta/per;
                printf("neiSta=%lf per=%lf ratio=%lf\n",neiSta,per,ratio);
                if ((double)rooms[merge_index].getNeighborStatistics()[current_room.getID()]/rooms[merge_index].getPerimeter() <= 0.1)
                    merge_rooms = false;
                if (merge_rooms == true)
                    break;
            }
        }

        if (merge_rooms == true)
        {
            std::cout << "d merge " << current_room.FastGetCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].FastGetCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
            FastMergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
            current_room_index = 0;
        }
        else
            current_room_index++;
    }
//    if (display_map == true)
//    {
//        cv::imshow("d", map_to_merge_rooms);
//        cv::waitKey();
//    }

    // e) largest room neighbor touches > 0.5 perimeter (happens often with furniture)
    for (int current_room_index = 0; current_room_index < rooms.size(); )
    {
        Room& current_room = rooms[current_room_index];
        bool merge_rooms = false;
        size_t merge_index = 0;

        const int max_border_neighbor_id = current_room.getNeighborWithLargestCommonBorder();
        if ((double)current_room.getNeighborStatistics()[max_border_neighbor_id]/current_room.getPerimeter() > 0.4)
        {
            // merge with that neighbor that shares the most neighboring pixels
            merge_rooms = determineRoomIndexFromRoomID(rooms, max_border_neighbor_id, merge_index);
        }

        if (merge_rooms == true)
        {
            std::cout << "e merge " << current_room.FastGetCenter() << ", id=" << current_room.getID() << " into " << rooms[merge_index].FastGetCenter() << ", id=" << rooms[merge_index].getID() << std::endl;
            FastMergeRoomPair(rooms, merge_index, current_room_index, map_to_merge_rooms, map_resolution_from_subscription);
            current_room_index = 0;
        }
        else
            current_room_index++;
    }

	
	 // 完成房间合并后，修正边界
   // AlignRoomBoundariesToRectangles(map_to_merge_rooms, rooms);

    time.stop();
    std::cout<<"mergeRoomPair take time"<<time.getElapsedTimeInSec()<<std::endl;
}
