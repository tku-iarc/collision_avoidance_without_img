cmake_minimum_required(VERSION 2.8.3)

project(mobile_table)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
)

foreach(dir config launch meshes urdf model_move)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
