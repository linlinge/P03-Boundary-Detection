#include <iostream>	
#include "PCLExtend.h"
#include "V3.hpp"
#include "DataMining.h"
#include "ImprovedLoop.h"
#include "EdgeDetection.h"
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <omp.h>

// Minor-Major Ratio, Central Limit Theorem
void MMR_CLT(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);

	// Step 2: calculate ratio=Minor/Major
	vector<double> mmr(cloud->points.size());
	vector<double> mmr_mean(cloud->points.size());
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);
		// Step 2-2: Establish cloud temp
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		for(int j=0;j<idx.size();j++){
			ctmp->points.push_back(cloud->points[idx[j]]);
		}
		// Step 2-3: get eigenvalue
		Eigen::Vector4f centroid;
		Eigen::Matrix3f covariance;
		pcl::compute3DCentroid(*ctmp, centroid);
		pcl::computeCovarianceMatrixNormalized(*ctmp, centroid, covariance);	
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);		
		Eigen::Vector3f eig_val = eigen_solver.eigenvalues();
		// Step 2-4: get ratio
		mmr[i]=eig_val(0)/eig_val(2);
	}

	// Step 3: calculate sample mean
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// step 3-1: find k-nearest neighbours
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		// step 3-2: calculate mean of mmr
		double cmean=0;
		for(int j=0;j<K;j++){
			cmean+=mmr[idx[j]];
		}
		cmean=cmean/K;
		// step 3-3: store cmean
		mmr_mean[i]=cmean;
	}

	// Step 4: calculate expectation,sigma for mmr_mean
	double E_mmr_mean=0;
	double sigma_mmr_mean=0;	
	for(int i=0;i<mmr_mean.size();i++){
		E_mmr_mean+=mmr_mean[i];
	}
	E_mmr_mean=E_mmr_mean/mmr_mean.size();

	for(int i=0;i<mmr_mean.size();i++){
		sigma_mmr_mean+=pow(mmr_mean[i]-E_mmr_mean,2);
	}
	sigma_mmr_mean=sqrt(sigma_mmr_mean/(mmr_mean.size()-1));

	// Step 5: detect outlier
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		double err=abs(mmr_mean[i]-E_mmr_mean);
		if(err>=3*sigma_mmr_mean){
			cloud->points[i].r=255;
			cloud->points[i].g=0;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("mmr.ply",*cloud);
}

// Radius, Central Limit Theorem
void R_CLT(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);

	// Step 2: calculate radius
	vector<double> r(cloud->points.size());
	vector<double> rmean(cloud->points.size());
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);

		// Step 2-2: Establish cloud temp
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		double dtmp=0;
		for(int j=0;j<dist.size();j++){
			dtmp+=dist[j];
		}
		r[i]=dtmp/K;
	}

	// Step 3: calculate sample mean
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// step 3-1: find k-nearest neighbours
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		// step 3-2: calculate mean of radius
		double mtmp=0;
		for(int j=0;j<K;j++){
			mtmp+=r[idx[j]];
		}
		mtmp=mtmp/K;
		// step 3-3: store rmean
		rmean[i]=mtmp;
	}

	// Step 4: calculate expectation,sigma for mmr_mean
	double E_rmean=0;
	double sigma_rmean=0;	
	for(int i=0;i<rmean.size();i++){
		E_rmean+=rmean[i];
	}
	E_rmean=E_rmean/rmean.size();

	for(int i=0;i<rmean.size();i++){
		sigma_rmean+=pow(rmean[i]-E_rmean,2);
	}
	sigma_rmean=sqrt(sigma_rmean/(rmean.size()-1));

	// Step 5: detect outlier
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		double err=abs(rmean[i]-E_rmean);
		if(err>=3*sigma_rmean){
			cloud->points[i].r=0;
			cloud->points[i].g=255;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("R_CLT.ply",*cloud);
}

double GaussianKernel(double u, int n)
{
	return 1/sqrt(0.2*M_PI)*exp(-pow(u,2)/0.2);
}

// Calculate entropy with kernel density estimation (KDE)
void EntropyWithKDE(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);
	vector<double> entropy;

	// Step 2: entropy
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		// pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);

		// Step 2-2: Calculate entropy
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		double dtmp=0;
		double h=0.5;
		for(int j=0;j<K;j++){
			dtmp+=GaussianKernel(dist[j]/h,K);
		}
		dtmp=dtmp/K/h;

		// Step 2-3: Rendering
		// cout<<dtmp<<endl;
		cloud->points[i].r=dtmp*255;
		cloud->points[i].g=0;
		cloud->points[i].b=0;
	}
}

int main(int argc,char** argv)
{
	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);	
	if (pcl::io::loadPLYFile<PointType>(argv[1], *cloud) == -1) 
	{
		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return (-1);
	}

	// DetectHoleEdge03_kNN(cloud);
	DetectHoleEdge03_Radius(cloud);  //*
	// DetectHoleEdge02_r(cloud);
	// MMR_CLT(cloud);
	// R_CLT(cloud);
	// DataMining dm(cloud);
	// dm.LOOP();
	// ImprovedLoop IL;
	// IL.Init(cloud);
	// IL.ILoop(32,0.8);
	// IL.StatisticCentreAndCentroid(cloud);
	// IL.StatisticMinorMajorRatio(cloud);
	// IL.EdgeDetection();
	// EntropyWithKDE(cloud);






	// pcl::io::savePLYFileBinary("MMR.ply",*cloud);


	// boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer")); 
	// // Set background
	// viewer->setBackgroundColor (1.0f, 1.0f, 1.0f);

	// //Set multi-color for point cloud
	// pcl::visualization::PointCloudColorHandlerRGBField<PointType> multi_color(cloud);
	
	// //Add the demostration point cloud data
	// viewer->addPointCloud<PointType> (cloud, multi_color, "cloud1");

	// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud1");

	// while(!viewer->wasStopped()){	
	// 	viewer->spin();
	// 	boost::this_thread::sleep (boost::posix_time::microseconds (10));
	// }
	cout<<"end"<<endl;
	return 0;
}
