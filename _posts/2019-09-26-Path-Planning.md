---
layout:     post
title:      Highway Path Planning
date:       2019-09-26
summary:    Trajectory planning for a car driving on a highway 
categories: SDC Trajectory_Planning Behavior_Planning Sensor_Fusion 
---

### Goals
The goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

| XY Coordinate | Frenet Coordinate |
|:----------------:|:--------------:|
![](/images/HighwayXY.png)|![](/images/FrenetS.png)|

## Why use Frenet Coordinates?

Representing curved road profiles in a cartesian coordinate system is sometimes cumbersome. Tasks like determining the lane in which a car is or if two cars are in the same lane , in cartesian coordinates, adds unneccessary complications. Projecting the ***(x,y)*** s to the Frenet coordinate space helps us over come this problem. The `s` dimentsion (longitudinal axis) increases with the length of the lane . The `d` dimension (lateral axis) is centered on the middle yellow line. In this project the lanes are 4m wide and the highway has three lanes. That implies, 

| d range | lane |
|:-------:|:-----:|
` 0 < d < 4 ` |left lane|
`4 < d < 8 ` | center lane |
` 8 < d < 12 ` | right lane |

![](/images/cartVsFre.png)

## Path planning approach

The tasks involved in this Highway driving challenge can be broken down into three parts

1. Localization and Sensor Fusion
2. Behavior Planning 
3. Trajectory planning

### Localization and Sensor Fusion 

The simulator feeds us the localization and sensor fusion data . Localization data gives us the precise location of the ego car. In our case we can get the ego car's (x,y) location, (s,d) location, its heading direction and the speed from the localization data. This can be found between lines 80-85 in `main.cpp`.

The sensor fusion data on the other hand gives us the information on the surrounding environment. If the onboard sensors (RADAR/LiDAR) senses vehicles around the ego car, their corresponding (s,d) and speed are extracted. With the current position and speed, we can estimate where the neighboring cars will be in the future as well. This can be found between lines 118-125 in `main.cpp`.

### Behavior Planning

A Finite State Machine approach has been followed in this project to define the behavior of the ego vehicle. The finite states that can exist are 

1. Keep Lane
2. Prepare Lane Change Left/Right
3. Lane change Left  
4. Lane change Right 

![](/images/FSM.png)

### Keep Lane
While the ego car is driving within the speed limit, if there is no car ahead in the lane that could cause a deceleration, this state makes the car stay in the current lane. 

### Prepare Lane Change Left/Right
If we sense a car ahead in our current lane, we first look to the left lane (provided we are not in the left most lane already) and check if there is a car adjacent to us or within 30m ahead or behind the ego car. 
Else if there is a car detected in the left lane we perform the same check on the right lane.

### Lane Change Left/Right 
If there is no car detected in the left lane, we change left by setting the current lane to left lane. Else if there is no car to the right of the ego car, we prepare to change to the right by setting the current lane to the right lane. 

If there is no room to change either left or right, we reduce the speed to avoid collision with the car ahead. 

Finally if there is no car ahead and we are below the speed limit, we gradually increase the speed to reach the allowable speed such that the jerk condition is not violated. 


## Trajectory Planning

The trajectory that the ego car needs to take to drive around the simulator track should be such that there are no sudden lateral or longitudinal accelerations that could case the ego car to violate the jerk constraint. To meet this criteria, this implementation follows the spline approach suggested in the walkthrough video. 

A spline fits a smooth curve through the points we desire to pass through. In this case we use 5 control points through which we want the spline to be fit. Of these 5 points, we reuse two points from the previous path so that the transition is smooth between the two splines. The other three points are chosen such that they are 30, 60 and 90 meters ahead of the ego car. This can be found between lines 189-250 in `main.cpp`.

The spline function can now be used to find all the points required for a smooth path curve. Assuming we want to fit a curve that takes us 30m ahead, given a reference velocity and the controller update rate of the simulator, we can get the number of points required and find their interpolated x and y points that can be fed back to the simulator. 


## Result 

The car successfully went around the simulator track without any collision or jerk violation. The final state is shown below. The link to the video can be found [here](https://youtu.be/0Rb1QsdWksU) 

![](/images/result.png)

## Improvements

There are several limitations due to the simplified Finite State Machine assumption used in this implementation. 

1. The ego car always choses the left lane first irrespective of the lane availability. The car doesnt have a way to identify the faster of the two lanes to make a lane change decision. 
2. The cost associated with lane change is only binary in this implementation ie. safe to change lane or not. A more descriptive cost function could help go around the track faster.
3. The sensor fusion data is used to check where the neighboring car is going to be in its own lane at a future time. Their heading directions are not used and hence if the tracked car does a sudden lane change we do not have a way to account for it. 
