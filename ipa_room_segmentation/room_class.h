#ifndef __ROOM_CLASS_H__
#define __ROOM_CLASS_H__

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <set>

#include <ipa_room_segmentation/contains.h>
struct room_rec_st
{
    int min_x;
    int min_y;
    int max_x;
    int max_y;
};
struct cv_Point_comp
{
    bool operator()(const cv::Point& lhs, const cv::Point& rhs) const
    {
        return ((lhs.y < rhs.y) || (lhs.y == rhs.y && lhs.x < rhs.x));
    }
};
//This is the class that represents a room. It has a ID-number, Points that belong to it and a list of neighbors.
class Room
{
public:

	// Struct that compares two given points and returns if the y-coordinate of the first is smaller or if they are equal if the
	// x-coordinate of the first is smaller. This is used for sets to easily store cv::Point objects and search for specific objects.
//	struct cv_Point_comp
//	{
//		bool operator()(const cv::Point& lhs, const cv::Point& rhs) const
//		{
//			return ((lhs.y < rhs.y) || (lhs.y == rhs.y && lhs.x < rhs.x));
//		}
//	};

	typedef std::set<cv::Point,cv_Point_comp> PointSet;


	Room(int id_of_room);

	// merges the provided room into this room
	void mergeRoom(Room& room_to_merge, double map_resolution);

    void FastMergeRoom(Room& room_to_merge);

	int insertMemberPoint(cv::Point new_member, double map_resolution);

	int insertMemberPoints(const std::vector<cv::Point>& new_members, double map_resolution);

	void addNeighbor(int new_neighbor_id);

	int addNeighborID(int new_neighbor_id);

	int getNeighborCount();

	std::map<int,int>& getNeighborStatistics();

	void getNeighborStatisticsInverse(std::map< int,int,std::greater<int> >& neighbor_room_statistics_inverse);

	int getNeighborWithLargestCommonBorder(bool exclude_wall=true);

	double getPerimeterRatioOfXLargestRooms(const int number_rooms);

	double getWallToPerimeterRatio();

	std::vector<int>& getNeighborIDs();

	double getArea();

	double getPerimeter();

	int getID() const;

	cv::Point getCenter();
    cv::Point FastGetCenter();
	const std::vector<cv::Point>& getMembers();

	int setRoomId(int new_value, cv::Mat& map);

    int FastSetRoomId(int new_value, cv::Mat& map);

	int setArea(double room_area);

    void mergeArea(double room_area);

    void mergeRoomRec(room_rec_st rec);

    int setPerimeter(double room_perimeter);

    void setRoomRec(room_rec_st rec);

    room_rec_st getRoomRec();

    void insertEdgePoint(cv::Point new_member);

    std::set<cv::Point, cv_Point_comp> getEdgePoint();
protected:
	int id_number_;

	std::vector<cv::Point> member_points_;

	std::vector<int> neighbor_room_ids_;

	std::map<int, int> neighbor_room_statistics_;		// maps from room labels of neighboring rooms to number of touching pixels of the respective neighboring room

	double room_area_;

	double room_perimeter_;

    room_rec_st room_rec;

    //std::vector<cv::Point> edge_points_;
    std::set<cv::Point, cv_Point_comp> edge_points_;
};


bool sortRoomsAscending(Room a, Room b);

#endif
