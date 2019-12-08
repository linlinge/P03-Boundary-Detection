#pragma once
#include "PCLExtend.h"
#ifndef PointN
#define PointN pcl::PointXYZRGBNormal
#endif
class Point
{
    public:
        int id_;
        double x_,y_,z_;
        double roll_,pitch_,yaw_;

        void RollPitchYaw(int id, double x, double y, double z){
            id_=id;
            double r=sqrt(pow(x,2)+pow(y,2)+pow(z,2));
            
            // X axis
            roll_=atan2(z,y);
            if(roll_<M_PI) roll_+=M_PI;

            // Y axis
            pitch_=atan2(x,z);
            if(pitch_<M_PI)  pitch_+=M_PI;

            // Z axis
            yaw_=atan2(y,x);
            if(yaw_<M_PI) yaw_+=M_PI;
        }
};

// Method 01: |Ni-Nj| > delta
void DetectHoleEdge01(pcl::PointCloud<PointN>::Ptr cloud);

// Method 02: Centroid != Sphere Centre
void DetectHoleEdge02_kNN(pcl::PointCloud<PointN>::Ptr cloud);
void DetectHoleEdge02_r(pcl::PointCloud<PointN>::Ptr cloud);

// Method 03: Direction Distribution
void DetectHoleEdge03_kNN(pcl::PointCloud<PointN>::Ptr cloud);
void DetectHoleEdge03_Radius(pcl::PointCloud<PointN>::Ptr cloud);
