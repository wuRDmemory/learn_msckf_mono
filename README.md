# Mono msckf

## image process
1. refers to VINS-Mono's image process.
2. uses grid cell to splits points.

## How to use
1. this project has some requirements
	a. glog
	b. gflags
	c. opencv
	d. absl(google)
	e. SuiteSparse
	f. eigen

2. when use ROS with this project, you can use this command to compiler project
```
catkin_make --pkg learn_msckf --cmake-args -DBUILD_TEST=OFF -DBUILD_DEBUG=OFF
```

