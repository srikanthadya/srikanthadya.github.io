---
layout:     post
title:      PID Controller
date:       2019-10-10
summary:    PID Controller for autonomous driving 
categories: SDC PID self-driving  
---
In an earlier project we had implemented a behavioral cloning algorithm in Python which would look at the image of the track ahead and predict what the steering angle should be to maintain the lane and drive around the simulator track. 

Another way to drive around the track would be to use the PID controller. The task of a PID controller is to ensure the vehicle is in the center of the lane by correcting for the error caused by deviating from the lane center. This deviation from the lane center is termed as **Cross Track Error (CTE)**. 

This correction **(Steering Angle)** is composed of three terms, the Proportional term, Integral Term and the Diffrential term and is expressed as below.


<a href="https://www.codecogs.com/eqnedit.php?latex=Steering&space;Angle&space;=&space;K_{p}*CTE&space;&plus;&space;K_{d}*\frac{d&space;(CTE)}{dt}&plus;&space;K_{i}*\int(CTE\&space;dt)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Steering&space;Angle&space;=&space;K_{p}*CTE&space;&plus;&space;K_{d}*\frac{d&space;(CTE)}{dt}&plus;&space;K_{i}*\int(CTE\&space;dt)" title="Steering Angle = K_{p}*CTE + K_{d}*\frac{d (CTE)}{dt}+ K_{i}*\int(CTE\ dt)" /></a>


## Proportional term

The proportional term tries to make to CTE reduce to zero by steering towards the center line. This term has the largest contribution toward the steering correction. But if used alone, this term will cause the vehicle to oscillate about the centerline. The best way to come up with an initial guess for the proportional constant Kp is to set the other two to zero and increase Kp untill the vehicle oscillates about the center line. 

## Diffrential term

In order to reduce the vehicle's oscillation explained above, the diffrential term is introduced such that the steering correction reduces as the CTE gets closer to zero. This ensures that as the vehicle nears the centerline, the steering correction also reduces proportionally this preventing an overshoot. 

## Integral term

A vehicle might have an inherent bias like a steering drift which will prevent it from achievng a zero CTE. To counter this, the CTE is accumulated and a correction proportional to the accumulated error is added.

![PID_image](/images/Twiddle1.png)

## Tuning the PID coefficients

As mentioned above the initial values of Kp, Kd and Ki can be obtained manually. Initially Kd an Ki are set to zero and Kp is increased untill the vehicle starts oscillating about the center line. Once this is achieved, Kd is increased untill the oscillations reduce and the vehicle follows the centerline. Ki on this track did not have much of an influence has been set to a very low value. 

## Twiddle

Once we have an initial value for the three parameters, we allow the twiddle algorithm to fine tune the coefficients further. It was noticed that the coefficients oscillated aboout the initial value and finally converged to the same values as the initial value. 

## Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/mxTvz2GgP0Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
