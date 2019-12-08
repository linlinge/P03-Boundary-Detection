#include "EdgeDetection.h"
#include <fstream>
using namespace std;

// Method 01: |Ni-Nj| > delta
void DetectHoleEdge01(pcl::PointCloud<PointN>::Ptr cloud)
{
	int K=32;
	pcl::search::KdTree<PointN>::Ptr kdtree(new pcl::search::KdTree<PointN>());
	kdtree->setInputCloud(cloud);
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		Eigen::MatrixXd Nc(3,1);
		Nc<<cloud->points[i].normal_x,cloud->points[i].normal_y,cloud->points[i].normal_z;
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		int count=0;
		for(int j=0;j<K;j++){			
			Eigen::MatrixXd Nn(3,1);
			Nn<<cloud->points[idx[j]].normal_x,cloud->points[idx[j]].normal_y,cloud->points[idx[j]].normal_z;
			Eigen::MatrixXd cos_arc=Nc.transpose()*Nn*1.0f/Nc.norm()/Nn.norm();
			double arc=acos(cos_arc(0));
			if(arc>M_PI/2.0){
				// cloud->points[idx[j]].r=255;
				// cloud->points[idx[j]].g=0;
				// cloud->points[idx[j]].b=0;
				count++;
			}
		}
		double ratio=count*1.0/K;
		if(ratio>0.8){
			cloud->points[i].r=255;
			cloud->points[i].g=0;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileASCII("EdgeDetect.ply",*cloud);
}

// Method 02: centroid != sphere centre
void DetectHoleEdge02_kNN(pcl::PointCloud<PointN>::Ptr cloud)
{
	int K=200;
	pcl::search::KdTree<PointN>::Ptr kdtree(new pcl::search::KdTree<PointN>());
	kdtree->setInputCloud(cloud);
	double mean_dist=ComputeMeanDistance(cloud);
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){		
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		double px,py,pz;
		px=py=pz=0;	
		for(int j=0;j<K;j++){			
			px=px+cloud->points[idx[j]].x;
			py=py+cloud->points[idx[j]].y;
			pz=pz+cloud->points[idx[j]].z;
		}
		px=px/K;
		py=py/K;
		pz=pz/K;
		double dist_centroid_centre=sqrt(pow(px-cloud->points[i].x,2)+pow(py-cloud->points[i].y,2)+pow(pz-cloud->points[i].z,2));
		if(dist_centroid_centre>3*mean_dist){
			cloud->points[i].r=0;
			cloud->points[i].g=255;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileASCII("EdgeDetect.ply",*cloud);
}

// Method 02: centroid != sphere centre
void DetectHoleEdge02_r(pcl::PointCloud<PointN>::Ptr cloud)
{	
	pcl::search::KdTree<PointN>::Ptr kdtree(new pcl::search::KdTree<PointN>());
	kdtree->setInputCloud(cloud);
	double mean_dist=ComputeMeanDistance(cloud);
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		vector<int> idx;
		vector<float> dist;
		kdtree->radiusSearch (cloud->points[i], 10*mean_dist, idx, dist);
		int K=idx.size();
		double px,py,pz;
		px=py=pz=0;
		for(int j=0;j<K;j++){
			px=px+cloud->points[idx[j]].x;
			py=py+cloud->points[idx[j]].y;
			pz=pz+cloud->points[idx[j]].z;
		}
		px=px/K;
		py=py/K;
		pz=pz/K;
		double dist_centroid_centre=sqrt(pow(px-cloud->points[i].x,2)+pow(py-cloud->points[i].y,2)+pow(pz-cloud->points[i].z,2));
		if(dist_centroid_centre>3*mean_dist){
			cloud->points[i].r=0;
			cloud->points[i].g=255;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileASCII("EdgeDetect.ply",*cloud);
}

// Method 03: Direction Distribution
void DetectHoleEdge03_Radius(pcl::PointCloud<PointN>::Ptr cloud)
{
    pcl::search::KdTree<PointN>::Ptr kdtree(new pcl::search::KdTree<PointN>());
	kdtree->setInputCloud(cloud);
	double mean_dist=ComputeMeanDistance(cloud);
    vector<int> boundary_id;
    vector<double> Kstatistical;
    Kstatistical.resize(cloud->points.size());

	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
        // Define
		vector<int> idx;
		vector<float> dist;
        double roll_gap,max_roll_gap;
        double pitch_gap,max_pitch_gap;
        double yaw_gap,max_yaw_gap;
        roll_gap=pitch_gap=yaw_gap=0;
        max_roll_gap=max_pitch_gap=max_yaw_gap=0;

        // Establish Kdtree
		kdtree->radiusSearch(cloud->points[i], 6*mean_dist, idx, dist);
		int K=idx.size();

        Kstatistical[i]=K;
        // temp cloud among radius
        vector<Point> ptmp;
        ptmp.resize(K);
        for(int j=1;j<K;j++){
            double delta_x=cloud->points[idx[j]].x-cloud->points[i].x;
            double delta_y=cloud->points[idx[j]].y-cloud->points[i].y;
            double delta_z=cloud->points[idx[j]].z-cloud->points[i].z;
            ptmp[j].RollPitchYaw(idx[j],delta_x,delta_y,delta_z);
        }

        // Calculate 01: max_roll_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.roll_<e2.roll_);});
        for(int j=1;j<K;j++){
            roll_gap=ptmp[j].roll_-ptmp[j-1].roll_;
            max_roll_gap=max_roll_gap>roll_gap ? max_roll_gap:roll_gap;
        }
        roll_gap=2*M_PI-(ptmp[K-1].roll_-ptmp[0].roll_);
        max_roll_gap=max_roll_gap>roll_gap ? max_roll_gap:roll_gap;

        // Calculate 02: max_pitch_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.pitch_<e2.pitch_);});
        for(int j=1;j<K;j++){
            pitch_gap=ptmp[j].pitch_-ptmp[j-1].pitch_;
            max_pitch_gap=max_pitch_gap>pitch_gap ? max_pitch_gap:pitch_gap;
        }
        pitch_gap=2*M_PI-(ptmp[K-1].pitch_-ptmp[0].pitch_);
        max_pitch_gap=max_pitch_gap>pitch_gap ? max_pitch_gap:pitch_gap;

        // Calculate 03: max_gamma_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.yaw_<e2.yaw_);});
        for(int j=1;j<K;j++){
            yaw_gap=ptmp[j].yaw_-ptmp[j-1].yaw_;
            max_yaw_gap=max_yaw_gap>yaw_gap ? max_yaw_gap:yaw_gap;
        }
        yaw_gap=2*M_PI-(ptmp[K-1].yaw_-ptmp[0].yaw_);
        max_yaw_gap=max_yaw_gap>yaw_gap ? max_yaw_gap:yaw_gap;

        // roll,pitch,yaw threshould
        if(max_roll_gap>M_PI/2.0 && max_pitch_gap>M_PI/2.0 && max_yaw_gap>M_PI/2.0){
            cloud->points[i].r=0;
            cloud->points[i].g=255;
            cloud->points[i].b=0;
            boundary_id.push_back(i);
        }
	}
    ofstream fout("BoundaryID.csv");
    for(int i=0;i<boundary_id.size();i++){
        fout<<boundary_id[i]<<endl;
    }
    fout.close();
	pcl::io::savePLYFileASCII("EdgeDetect.ply",*cloud);
}


// Method 03: Direction Distribution (kNN)
void DetectHoleEdge03_kNN(pcl::PointCloud<PointN>::Ptr cloud)
{
    pcl::search::KdTree<PointN>::Ptr kdtree(new pcl::search::KdTree<PointN>());
	kdtree->setInputCloud(cloud);
	double mean_dist=ComputeMeanDistance(cloud);
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
        // Define
		vector<int> idx;
		vector<float> dist;
        double roll_gap,max_roll_gap;
        double pitch_gap,max_pitch_gap;
        double yaw_gap,max_yaw_gap;
        roll_gap=pitch_gap=yaw_gap=0;
        max_roll_gap=max_pitch_gap=max_yaw_gap=0;

        // Establish Kdtree
        int K=40;
		kdtree->nearestKSearch (cloud->points[i], K, idx, dist);

        // kNN temp cloud
        vector<Point> ptmp;
        ptmp.resize(K);
        for(int j=1;j<K;j++){
            double delta_x=cloud->points[idx[j]].x-cloud->points[i].x;
            double delta_y=cloud->points[idx[j]].y-cloud->points[i].y;
            double delta_z=cloud->points[idx[j]].z-cloud->points[i].z;
            ptmp[j].RollPitchYaw(idx[j],delta_x,delta_y,delta_z);
        }            

        // Calculate 01: max_roll_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.roll_<e2.roll_);});
        for(int j=1;j<K;j++){
            roll_gap=ptmp[j].roll_-ptmp[j-1].roll_;
            max_roll_gap=max_roll_gap>roll_gap ? max_roll_gap:roll_gap;
        }
        roll_gap=2*M_PI-(ptmp[K-1].roll_-ptmp[0].roll_);
        max_roll_gap=max_roll_gap>roll_gap ? max_roll_gap:roll_gap;

        // Calculate 02: max_pitch_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.pitch_<e2.pitch_);});
        for(int j=1;j<K;j++){
            pitch_gap=ptmp[j].pitch_-ptmp[j-1].pitch_;
            max_pitch_gap=max_pitch_gap>pitch_gap ? max_pitch_gap:pitch_gap;
        }
        pitch_gap=2*M_PI-(ptmp[K-1].pitch_-ptmp[0].pitch_);
        max_pitch_gap=max_pitch_gap>pitch_gap ? max_pitch_gap:pitch_gap;

        // Calculate 03: max_gamma_gap
        sort(ptmp.begin(),ptmp.end(),[](Point& e1, Point& e2){ return (e1.yaw_<e2.yaw_);});
        for(int j=1;j<K;j++){
            yaw_gap=ptmp[j].yaw_-ptmp[j-1].yaw_;
            max_yaw_gap=max_yaw_gap>yaw_gap ? max_yaw_gap:yaw_gap;
        }
        yaw_gap=2*M_PI-(ptmp[K-1].yaw_-ptmp[0].yaw_);
        max_yaw_gap=max_yaw_gap>yaw_gap ? max_yaw_gap:yaw_gap;

        // roll,pitch,yaw threshould
        if((max_roll_gap>M_PI/2.0) && (max_pitch_gap>M_PI/2.0 || max_yaw_gap>M_PI/2.0)){
            // if(max_pitch_gap>M_PI/2.0 && max_yaw_gap>M_PI/2.0){
                
            // }
            // else{
            //     cloud->points[i].r=255;
            //     cloud->points[i].g=0;
            //     cloud->points[i].b=0;
            // }

            
            cloud->points[i].r=255;
            cloud->points[i].g=0;
            cloud->points[i].b=0;
        }
	}
	pcl::io::savePLYFileASCII("EdgeDetect.ply",*cloud);
}


// Method 04: pcl api (based on normal estimation)
void EdgeDetection(pcl::PointCloud<PointType>::Ptr cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Boundary> boundaries;
    pcl::BoundaryEstimation<PointType,pcl::Normal,pcl::Boundary> est;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    pcl::NormalEstimation<PointType,pcl::Normal> normEst;  //其中pcl::PointXYZ表示输入类型数据，pcl::Normal表示输出类型,且pcl::Normal前三项是法向，最后一项是曲率
    normEst.setInputCloud(cloud);
    normEst.setSearchMethod(tree);
    // normEst.setRadiusSearch(2);  //法向估计的半径
    normEst.setKSearch(50);  //法向估计的点数
    normEst.compute(*normals);
    cout<<"normal size is "<< normals->size()<<endl;

    //normal_est.setViewPoint(0,0,0); //这个应该会使法向一致
    est.setInputCloud(cloud);
    est.setInputNormals(normals);
    est.setAngleThreshold(M_PI/4.0);
    //   est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
    est.setSearchMethod (tree);
    est.setKSearch(100);  //一般这里的数值越高，最终边界识别的精度越好
    //  est.setRadiusSearch(everagedistance);  //搜索半径
    est.compute (boundaries);

    //  pcl::PointCloud<pcl::PointXYZ> boundPoints;
    pcl::PointCloud<PointType>::Ptr boundPoints (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType> noBoundPoints;
    int countBoundaries = 0;
    for(int i=0; i<cloud->size(); i++){
        int x = (boundaries.points[i].boundary_point);
        int a = static_cast<int>(x); //该函数的功能是强制类型转换
        if(a == 1)
        {
            //  boundPoints.push_back(cloud->points[i]);
            // (*boundPoints).push_back(cloud_->points[i]);
            countBoundaries++;
            cloud->points[i].r=255;
            cloud->points[i].g=0;
            cloud->points[i].b=0;
        }
        else
            noBoundPoints.push_back(cloud->points[i]);
    }
    std::cout<<"boudary size is：" <<countBoundaries <<std::endl;
    pcl::io::savePLYFileBinary("EdgeDetect.ply",*cloud);
}