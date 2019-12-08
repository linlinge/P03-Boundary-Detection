#include "ImprovedLoop.h"
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>

void ImprovedLoop::StatisticCentreAndCentroid(pcl::PointCloud<PointType>::Ptr cloud)
{    
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    double mean_dist=ComputeMeanDistance(cloud);
    vector<double> stcc;
    stcc.resize(cloud->points.size());
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx;
        vector<float> dist;
        kdtree->radiusSearch(cloud->points[i], 8*mean_dist, idx, dist);

        // Calculate centroid
        double cx,cy,cz;
        cx=cy=cz=0;
        for(int j=0;j<idx.size();j++){
            cx+=cloud->points[idx[j]].x;
            cy+=cloud->points[idx[j]].y;
            cz+=cloud->points[idx[j]].z;
        }
        cx=cx/idx.size();
        cy=cy/idx.size();
        cz=cz/idx.size();

        // calculate distance
        stcc[i]=sqrt(pow(cx-cloud->points[i].x,2)+pow(cy-cloud->points[i].y,2)+pow(cz-cloud->points[i].z,2));
    }

    ofstream fout("stcc.csv");
    for(int i=0;i<cloud->points.size();i++){
        fout<<stcc[i]<<endl;
    }
    fout.close();
}






double ImprovedLoop::erf(double x)
{
	double a1=0.278393;
	double a2=0.230389;
	double a3=0.000972;
	double a4=0.078108;
	double m=1+a1*x+a2*pow(x,2)+a3*pow(x,3)+a4*pow(x,4);
	return (1-1.0/pow(m,4));
}

void ImprovedLoop::Init(pcl::PointCloud<PointType>::Ptr cloud)
{
	// set cloud
	cloud_=cloud;
	// establish kdtree_
	kdtree_=pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>());
	kdtree_->setInputCloud(cloud_);
}

void ImprovedLoop::ILoop(int k,double threshould)
{
	// Resize Scores
	scores_.resize(cloud_->points.size());
	sigma_.resize(cloud_->points.size());
	plof_.resize(cloud_->points.size());

	// Step 01: Calculate sigma
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		// step 1-1: find k-nearest neighours
		vector<int> idx(k+1);
		vector<float> dist(k+1);
		kdtree_->nearestKSearch(cloud_->points[i], k+1, idx, dist);
        pcl::PointCloud<PointType>::Ptr ptmp(new pcl::PointCloud<PointType>);
        for(int j=0;j<idx.size();j++){
            ptmp->points.push_back(cloud_->points[idx[j]]);
        }

        // step 1-2: get projection vector
        EvalAndEvec vv;
        vv.MajorMinor(ptmp);
        // cout<<vv.minor_vec_<<endl;
        Eigen::Vector3f v1;
        v1=vv.minor_vec_;

        // step 1-3: calculate sum	
		double sum=0;
        PointType p1=cloud_->points[i];
		for(int j=1;j<k+1;j++){
            PointType p2=cloud_->points[idx[j]];
            
            Eigen::Vector3f v2;
            v2<<p2.x,p2.y,p2.z;
            // cout<<v1*v2<<endl;
            auto prj=v1.transpose()*v2/v1.norm();
			sum+=abs(prj(0));
		}
		sum=sum/k;
		sigma_[i]=sqrt(sum);
	}
	
	// Step 02: calculate mean
	double mean=0;
	// #pragma omp parallel for
	for (int i = 0; i < cloud_->points.size(); i++){        
        vector<int> pointIdxNKNSearch(k+1);
		vector<float> pointNKNSquaredDistance(k+1);
		kdtree_->nearestKSearch (cloud_->points[i], k+1, pointIdxNKNSearch, pointNKNSquaredDistance);
        double sum = 0;
        for (int j = 1; j < k+1; j++)
          sum += sigma_[pointIdxNKNSearch[j]];
        sum /= k;
        plof_[i] = sigma_[i] / sum  - 1.0f;
        mean += plof_[i] * plof_[i];
    }
	mean=mean/cloud_->points.size();
	mean=sqrt(mean);

	// Step 03: caculate score
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		double value = plof_[i] / (mean * sqrt(2.0f));
        double dem = 1.0 + 0.278393 * value;
        dem += 0.230389 * value * value;
        dem += 0.000972 * value * value * value;
        dem += 0.078108 * value * value * value * value;
        double op = std::max(0.0, 1.0 - 1.0 / dem);
        scores_[i] = op;
	}

	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		if(scores_[i]>threshould){
			cloud_->points[i].r=255;
			cloud_->points[i].g=0;
			cloud_->points[i].b=0;
		}
	}
}