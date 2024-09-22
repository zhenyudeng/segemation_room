#include <ipa_room_segmentation/wavefront_region_growing.h>

// spreading image is supposed to be of type CV_32SC1
void wavefrontRegionGrowing(cv::Mat& image)
{
	//This function spreads the colored regions of the given map to the neighboring white pixels
	if (image.type()!=CV_32SC1)
	{
		std::cout << "Error: wavefrontRegionGrowing: provided image is not of type CV_32SC1." << std::endl;
		return;
	}

	cv::Mat spreading_map = image.clone();
	bool finished = false;
	while (finished == false)
	{
		finished = true;
		for (int row = 1; row < spreading_map.rows-1; ++row)
		{
			for (int column = 1; column < spreading_map.cols-1; ++column)
			{
				if (spreading_map.at<int>(row, column) > 65279)		// unassigned pixels
				{
					//check 3x3 area around white pixel for fillcolour, if filled Pixel around fill white pixel with that colour
					bool set_value = false;
					for (int row_counter = -1; row_counter <= 1 && set_value==false; ++row_counter)
					{
						for (int column_counter = -1; column_counter <= 1 && set_value==false; ++column_counter)
						{
							int value = image.at<int>(row + row_counter, column + column_counter);
							if (value != 0 && value <= 65279)
							{
								spreading_map.at<int>(row, column) = value;
								set_value = true;
								finished = false;	// keep on iterating the wavefront propagation until no more changes occur
							}
						}
					}
				}
			}
		}
		image = spreading_map.clone();
	}
}
void wavefrontRegionGrowingAndGetRec(cv::Mat& image, std::map<int, size_t> label_vector_index_codebook,
                                     std::vector<Room>& rooms)
{

    //This function spreads the colored regions of the given map to the neighboring white pixels
    if (image.type()!=CV_32SC1)
    {
        std::cout << "Error: wavefrontRegionGrowing: provided image is not of type CV_32SC1." << std::endl;
        return;
    }
    std::vector<int> min_x_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_x_value_of_the_room(label_vector_index_codebook.size(), 0);
    std::vector<int> min_y_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_y_value_of_the_room(label_vector_index_codebook.size(), 0);

    cv::Mat spreading_map = image.clone();
    bool finished = false;
    int cnt=0;
    while (finished == false)
    {
        finished = true;
        int set_cnt=0;
        for (int row = 1; row < spreading_map.rows-1; ++row)
        {
            for (int column = 1; column < spreading_map.cols-1; ++column)
            {
                if (spreading_map.at<int>(row, column) > 65279)		// unassigned pixels
                {
                    //check 3x3 area around white pixel for fillcolour, if filled Pixel around fill white pixel with that colour
                    bool set_value = false;
                    for (int row_counter = -1; row_counter <= 1 && set_value==false; ++row_counter)
                    {
                        for (int column_counter = -1; column_counter <= 1 && set_value==false; ++column_counter)
                        {
                            int value = image.at<int>(row + row_counter, column + column_counter);
                            if (value != 0 && value <= 65279)
                            {
                                set_cnt++;
                                spreading_map.at<int>(row, column) = value;
                                const int index = label_vector_index_codebook[value];
                                //!注意column对应x
                                min_x_value_of_the_room[index] = std::min(column, min_x_value_of_the_room[index]);
                                max_x_value_of_the_room[index] = std::max(column, max_x_value_of_the_room[index]);
                                max_y_value_of_the_room[index] = std::max(row, max_y_value_of_the_room[index]);
                                min_y_value_of_the_room[index] = std::min(row, min_y_value_of_the_room[index]);
                                set_value = true;
                                finished = false;	// keep on iterating the wavefront propagation until no more changes occur
                            }
                        }
                    }
                }
            }
        }
        cnt++;
        //printf("cnt=%d set_cnt=%d\n",cnt,set_cnt);
        image = spreading_map.clone();
    }
    for(int i=0;i<(int)rooms.size();i++)
    {
        int label=rooms[i].getID();
        const int index = label_vector_index_codebook[label];
        room_rec_st rec;

        rec.min_x=min_x_value_of_the_room[index];
        rec.min_y=min_y_value_of_the_room[index];
        rec.max_x=max_x_value_of_the_room[index];
        rec.max_y=max_y_value_of_the_room[index];
//        if(rec.min_x<0)
//        {
//            rec.min_x=0;
//        }
//        if(rec.min_y<0)
//        {
//            rec.min_y=0;
//        }
//        if(rec.max_x>=image.cols)
//        {
//            rec.max_x=image.cols-1;
//        }
//        if(rec.max_y>=image.rows)
//        {
//            rec.max_y=image.rows-1;
//        }
        rooms[i].setRoomRec(rec);

//        printf("label=%d,index=%d min_x=%d,max_x=%d min_y=%d max_y=%d\n",label,index,
//                min_x_value_of_the_room[index],max_x_value_of_the_room[index]
//               ,min_y_value_of_the_room[index],max_y_value_of_the_room[index]);
    }


}
