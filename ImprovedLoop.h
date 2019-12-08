#pragma once
#include "PCLExtend.h"
#include <Eigen/Dense>
#include <iostream>
using namespace std;
class Result
{
    public:
        int id_;        
        double major_;
        double minor_;
        double ratio_;
};

class ImprovedLoop
{
    public:
        pcl::PointCloud<PointType>::Ptr cloud_;
        pcl::search::KdTree<PointType>::Ptr kdtree_;

        // Estimate multivariable normal distribution parameters
        void StatisticCentreAndCentroid(pcl::PointCloud<PointType>::Ptr cloud);

        
        // ILOOP Parameters
		vector<double> scores_;
		vector<double> sigma_;
		vector<double> plof_;
        double erf(double x);
        void Init(pcl::PointCloud<PointType>::Ptr cloud);
        void ILoop(int k,double threshould);
};