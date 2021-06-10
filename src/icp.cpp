#include <Eigen/Dense>
#include <Eigen/SVD>
#include <random>
#include <iostream>
#include <math.h>
#include <time.h>
#include <igl/boundary_loop.h>
#include <igl/bounding_box.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/fit_plane.h>
#include "nanoflann.hpp"
#include "icp.h"

// Here I give as input the vertices and the amount of subsample I want to take out of it.
// Return the new V with the x% less number of points
// Subsampling is done randomly as explained in the report.
Eigen::MatrixXd ICP::GetSubsample(Eigen::MatrixXd vertices, double subsamplePercentage)
{
    std::vector<int> index;
    for (size_t i = 0; i < vertices.rows(); i++) 
    {
        if (rand() / double(RAND_MAX) >= subsamplePercentage/100) 
        {
            index.push_back(i);
        }
    }

    Eigen::MatrixXd newV(index.size(), 3);
    newV.setZero();

    for (size_t i = 0; i < index.size(); i++) 
    {
        newV.row(i) = vertices.row(index[i]);
    }

    return newV;
}

// Input Vertices which contains all the vertices and output Normals the normals for vertices
Eigen::MatrixXd ICP::GetNormals(Eigen::MatrixXd targetVertices)
{

    // p <-matched q<-to process
    Eigen::MatrixXd normals;
    normals.resize(targetVertices.rows(),targetVertices.cols());
    normals.setZero();

    const size_t resultNumber = 20;
    const size_t maxLeaf = 50;

    Eigen::MatrixXd center(1,3);
    center = targetVertices.colwise().sum()/ double(targetVertices.rows());

    // Using nanoflann to create a KD tree 
    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(targetVertices, maxLeaf);
    kd_tree_index.index->buildIndex();

    // For each vertex I create a query object to find the closest vertex and assignt it for the output
    for (size_t v=0; v<normals.rows(); v++) 
    {
        Eigen::RowVector3d query_vertex = targetVertices.row(v);

        std::vector<size_t> indexes(resultNumber);
        std::vector<double> dists_sqr(resultNumber);

        nanoflann::KNNResultSet<double> result(resultNumber);
        result.init(indexes.data(), dists_sqr.data());
        kd_tree_index.index->findNeighbors(result, query_vertex.data(), nanoflann::SearchParams(maxLeaf));

        Eigen::MatrixXd foundedVert(resultNumber, 3);
        for (size_t i = 0; i < resultNumber; i++){
            foundedVert.row(i) = targetVertices.row(indexes[i]);
        }

        Eigen::RowVector3d N, C;
        igl::fit_plane(foundedVert, N, C);

        normals.row(v) = N;

        // Looping through the normals to check their direction. If its wrong. inverse it.
        if((center(0,0)-targetVertices(v,0)) * normals(v,0) + (center(0,1)-targetVertices(v,1)) * normals(v,1) + (center(0,2)-targetVertices(v,2)) * normals(v,2) > 0) 
        {
            normals.row(v) = -normals.row(v);
        }
    }

    return normals;
}

std::pair<Eigen::MatrixXi, Eigen::MatrixXi> ICP::FindNonOverlappingFaces(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, Eigen::MatrixXi Ftoprocess)
{

    // output matrix
    std::vector<int> vDistant;
    std::vector<Eigen::Vector3i> F_raw;
    Eigen::MatrixXi fNonOverlap(0,3);
    Eigen::MatrixXi fOverlap(0,3);
    Eigen::Vector3i empty (-1,-1,-1);
    
    const size_t resultNumber = 1;
    const size_t maxLeaf = 10;
    const double threshold = 0.00001;

    // Using nanoflann to create a KD tree 
    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(targetVertices, maxLeaf);
    kd_tree_index.index->buildIndex();

    // For each vertex find closest vertex and check if it is distant
    for (size_t v=0; v<vertices.rows(); v++)
    {
        Eigen::RowVector3d query_vertex = vertices.row(v);

        std::vector<size_t> indexes(resultNumber);
        std::vector<double> dists_sqr(resultNumber);

        nanoflann::KNNResultSet<double> result(resultNumber);
        result.init(indexes.data(), dists_sqr.data());
        kd_tree_index.index->findNeighbors(result, query_vertex.data(), nanoflann::SearchParams(maxLeaf));

        // Check if it is distant
        if (dists_sqr[0] > threshold)
        {
            vDistant.push_back(v);
        }
    }

    for (size_t f=0; f<Ftoprocess.rows(); f++)
    {
        F_raw.push_back(Ftoprocess.row(f));
    }

    // non overlapping face list
    for (size_t f=0; f<F_raw.size(); f++)
    {
        for (size_t i=0; i < vDistant.size(); i ++)
        {
            for (size_t k=0; k<F_raw[f].size(); k++)
            {
                if (F_raw[f][k] == vDistant[i])
                {
                    fNonOverlap.conservativeResize(fNonOverlap.rows()+1, 3);
                    fNonOverlap.row(fNonOverlap.rows()-1) = Ftoprocess.row(f);
                    F_raw[f] = empty;
                    
                    break;
                }
            }
        }
    }

    // overlapping face list
    for (size_t i=0; i <F_raw.size(); i++)
    {
        if (F_raw[i] != empty)
        {
            fOverlap.conservativeResize(fOverlap.rows()+1, 3);
            fOverlap.row(fOverlap.rows()-1) = F_raw[i];
        }
    }

    return std::pair<Eigen::MatrixXi, Eigen::MatrixXi>(fOverlap, fNonOverlap);
}


Eigen::MatrixXd ICP::Rotate(Eigen::MatrixXd vertices, double x, double y, double z)
{
    // creating the output
    Eigen::MatrixXd outputVertices;
    outputVertices.resize(vertices.rows(), vertices.cols());
    outputVertices.setZero();
    
    // rotation matrix
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(x * M_PI/180, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(y * M_PI/180, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(z * M_PI/180, Eigen::Vector3d::UnitZ());
    
    // Rotate vertices
    outputVertices = vertices * R;
    
    return outputVertices;
}


Eigen::MatrixXd ICP::AddNoise(Eigen::MatrixXd vertices, double std)
{
    
    // Initialise output matrix
    Eigen::MatrixXd outputVertices;
    outputVertices.resize(vertices.rows(), vertices.cols());
    outputVertices.setZero();

    Eigen::MatrixXd boundingV;
    Eigen::MatrixXi boundingF;

    igl::bounding_box(vertices, boundingV, boundingF);
    double noise_scale_x = abs(boundingV.row(0).x()-boundingV.row(4).x());
    double noise_scale_y = abs(boundingV.row(0).y()-boundingV.row(2).y());
    double noise_scale_z = abs(boundingV.row(0).z()-boundingV.row(1).z());

    std::default_random_engine rnd;
    std::normal_distribution<double> gaussian(0.0, std);
    
    // Add noise to the vertices in all axis
    for (int i=0; i<outputVertices.rows(); i++)
    {
        double x =gaussian(rnd)/(10000*noise_scale_x);
        double y =gaussian(rnd)/(10000*noise_scale_y);
        double z =gaussian(rnd)/(10000*noise_scale_z);
        Eigen::RowVector3d noise(x,y,z);
        outputVertices.row(i) = vertices.row(i) + noise;
    }
    return outputVertices;
}


Eigen::MatrixXd ICP::FindBestStartRotation(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices)
{
    std::vector<Eigen::MatrixXd> VerticesRotList;
    std::vector<double> distanceList;

    // Applying rotations for each axis of 0(360), 120 and 240 separatly
    for (int x = 0; x < 3; x++)
    {
        for (int y = 0; y < 3; y++)
        {
            for (int z = 0; z < 3; z ++)
            {
                Eigen::MatrixXd V_rotated = Rotate(vertices, x*120, y*120, z*120);
                VerticesRotList.push_back(V_rotated);
                Eigen::MatrixXd matchedVertices = FindCorrespondences(targetVertices, V_rotated).first;
                Eigen::RowVector3d center_matched = matchedVertices.colwise().sum()/matchedVertices.rows();
                Eigen::RowVector3d center_rotated = V_rotated.colwise().sum()/V_rotated.rows();
                double distance = (center_matched-center_rotated).norm();
                distanceList.push_back(distance);
            }
        }
    }

    // Find smallest euclidean distance
    size_t min = std::min_element(distanceList.begin(),distanceList.end()) - distanceList.begin();
    return VerticesRotList[min];
}


// Input is the M1(V1) and M2(V2)
// Return will be the matched points but in m2(v2) with rejection
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ICP::FindCorrespondences(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices)
{

    // output matrix
    Eigen::MatrixXd outputVertices;
    std::vector<double> distances, rawDistances;
    outputVertices.resize(vertices.rows(), vertices.cols());
    outputVertices.setZero();

    std::vector<int> newIndex;

    const double k = 2.0;
    const size_t resultNumber = 1;
    const size_t maxLeaf = 20;

    double medianDist; // median distance

    // KD tree with nanoflann
    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(targetVertices, maxLeaf);
    kd_tree_index.index->buildIndex();

    // For each vertex find colosest vertex and to the outputVertices
    for (size_t v=0; v<outputVertices.rows(); v++)
    {
        Eigen::RowVector3d query_vertex = vertices.row(v);

        std::vector<size_t> indexes(resultNumber);
        std::vector<double> dists_sqr(resultNumber);

        nanoflann::KNNResultSet<double> result(resultNumber);
        result.init(indexes.data(), dists_sqr.data());
        kd_tree_index.index->findNeighbors(result, query_vertex.data(), nanoflann::SearchParams(maxLeaf));

        outputVertices.row(v) = targetVertices.row(indexes[0]);
        distances.push_back(dists_sqr[0]);
    }

    rawDistances = distances;

    // Find median distance value
    std::sort(distances.begin(),distances.end());

    if (distances.size() % 2 == 0)
    {
        medianDist = (distances[distances.size()/2-1]+distances[distances.size()/2])/2;
    }
    else
    {
        medianDist = distances[distances.size()/2];
    }

    for (int i = 0; i < rawDistances.size(); i++)
    {
        if (rawDistances[i] <= k * medianDist)
        {
            newIndex.push_back(i);
        }
        else
        {
            // ignore distant vertex
        }
    }

    Eigen::MatrixXd refinedOutput(newIndex.size(), 3);
    Eigen::MatrixXd refinedRaw(newIndex.size(), 3);

    for (int i = 0; i < newIndex.size(); i++)
    {
        refinedOutput.row(i) = outputVertices.row(newIndex[i]);
        refinedRaw.row(i) = vertices.row(newIndex[i]);
    }

    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>(refinedOutput, refinedRaw);
}


// input is M1, M2 and normals
// outwult matchedM1s. normals matched and m2 with rejections
std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::MatrixXd> ICP::FindCorrespondencesNormalBased(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, Eigen::MatrixXd Ntarget)
{
    //output matrix
    Eigen::MatrixXd outputVertices, outputN;
    std::vector<double> distances, rawDistance;
    outputVertices.resize(vertices.rows(), vertices.cols());
    outputVertices.setZero();

    outputN.resize(vertices.rows(), vertices.cols());
    outputN.setZero();

    std::vector<int> newIndex;

    const double k = 1.0;
    const size_t resultNumber = 1;
    const size_t maxLeaf = 20;

    double medianDist; // median distance

    // kd tree with nanoflann
    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_index(targetVertices, maxLeaf);
    kd_tree_index.index->buildIndex();

    // For each vertex find closest verted and assign to output
    for (size_t v=0; v<outputVertices.rows(); v++)
    {
        Eigen::RowVector3d query_vertex = vertices.row(v);

        std::vector<size_t> indexes(resultNumber);
        std::vector<double> dists_sqr(resultNumber);

        nanoflann::KNNResultSet<double> result(resultNumber);
        result.init(indexes.data(), dists_sqr.data());
        kd_tree_index.index->findNeighbors(result, query_vertex.data(), nanoflann::SearchParams(maxLeaf));

        outputVertices.row(v) = targetVertices.row(indexes[0]);
        outputN.row(v) = Ntarget.row(indexes[0]);
        distances.push_back(dists_sqr[0]);
    }

    rawDistance = distances;

    // median distance value
    std::sort(distances.begin(),distances.end());

    if (distances.size() % 2 == 0)
    {
        medianDist = (distances[distances.size()/2-1]+distances[distances.size()/2])/2;
    }
    else
    {
        medianDist = distances[distances.size()/2];
    }

    for (int i = 0; i < rawDistance.size(); i++)
    {
        if (rawDistance[i] <= k * medianDist)
        {
            newIndex.push_back(i);
        }
        else
        {
            // Distant vertex, ignore.
        }
    }

    Eigen::MatrixXd refinedOutput(newIndex.size(), 3);
    Eigen::MatrixXd refinedRaw(newIndex.size(), 3);
    Eigen::MatrixXd refinedOutNorm(newIndex.size(), 3);

    for (int i = 0; i < newIndex.size(); i++)
    {
        refinedOutput.row(i) = outputVertices.row(newIndex[i]);
        refinedRaw.row(i) = vertices.row(newIndex[i]);
        refinedOutNorm.row(i) = outputN.row(newIndex[i]);
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> refinedTarget (refinedOutput, refinedOutNorm);

    return std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::MatrixXd> (refinedTarget, refinedRaw);
}


// as input is m1 and M2 with mathced points
// output will be the R and T based on the min(R,t) and min(R) formulas ase seen in lectures
std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> ICP::EstimateRigidTransform(Eigen::MatrixXd matchedVertices, Eigen::MatrixXd vertices)
{

    std::pair<Eigen::Matrix3d, Eigen::RowVector3d> transform;

    Eigen::RowVector3d pBar = matchedVertices.colwise().mean();
    Eigen::RowVector3d qBar = vertices.colwise().mean();

    Eigen::MatrixXd pTilda = matchedVertices.rowwise() - pBar;
    Eigen::MatrixXd qTilda = vertices.rowwise() - qBar;

    // initialising A
    Eigen::Matrix3d A;
    A.setZero();

    for (size_t i=0; i<matchedVertices.rows(); i++)
    {
        Eigen::Vector3d p_i = pTilda.row(i);
        Eigen::Vector3d q_i = qTilda.row(i);

        A += q_i * p_i.transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();
    Eigen::RowVector3d T = pBar - (R * qBar.transpose()).transpose();

    transform.first = R;
    transform.second = T;

    double error = GetErrorMetric(matchedVertices, ApplyRigidTransform(vertices,transform));

    return std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> (error, transform);

}

// as input is m1 and M2 with mathced points and normals
// output will be the R and T
std::pair<Eigen::Matrix3d, Eigen::RowVector3d> ICP::EstimateRigidTransformNormalBased(Eigen::MatrixXd matchedVertices, Eigen::MatrixXd vertices, Eigen::MatrixXd NtoProcess)
{

    Eigen::MatrixXd A (matchedVertices.rows(), 6);
    Eigen::MatrixXd b (matchedVertices.rows(), 1);

    for (size_t i = 0; i < matchedVertices.rows(); i++)
    {
        Eigen::RowVector3d N = NtoProcess.row(i);
        Eigen::RowVector3d S = vertices.row(i);
        Eigen::RowVector3d D = matchedVertices.row(i);

        A(i,0) = N.z()*D.y()-N.y()*D.z();
        A(i,1) = N.x()*D.z()-N.z()*D.x();
        A(i,2) = N.y()*D.x()-N.x()*D.y();
        A(i,3) = N.x();
        A(i,4) = N.y();
        A(i,5) = N.z();

        b(i) = N.x()*D.x() + N.y()*D.y() + N.z()*D.z() - N.x()*S.x() - N.y()*S.y() - N.z()*S.z();
    }

    // x = (alpha beta gamma t_x t_y t_z).T
    Eigen::MatrixXd x = ((A.transpose() * A).inverse()) * (A.transpose()) * b;

    // Rigid transform R and T
    Eigen::Matrix3d R;
    R.setZero();

    double sin_alpha = sin(x(0));
    double cos_alpha = cos(x(0));
    double sin_beta = sin(x(1));
    double cos_beta = cos(x(1));
    double sin_gamma = sin(x(2));
    double cos_gamma = cos(x(2));

    R(0,0) = cos_gamma * cos_beta;
    R(0,1) = -sin_gamma * cos_alpha + cos_gamma * sin_beta * sin_alpha;
    R(0,2) = sin_gamma * sin_alpha + cos_gamma * sin_beta * cos_alpha;

    R(1,0) = sin_gamma * cos_beta;
    R(1,1) = cos_gamma * cos_alpha + sin_gamma * sin_beta * sin_alpha;
    R(1,2) = -cos_gamma * sin_alpha + sin_gamma * sin_beta * cos_alpha;

    R(2,0) = -sin_beta;
    R(2,1) = cos_beta * sin_alpha;
    R(2,2) = cos_beta * cos_alpha;

    Eigen::RowVector3d T(x(3),x(4),x(5));

    return std::pair<Eigen::Matrix3d, Eigen::RowVector3d>(R,T);
}


// input will be M2 and R and T which will be applied to M2
// as output we get the new M2
Eigen::MatrixXd ICP::ApplyRigidTransform(Eigen::MatrixXd vertices, std::pair<Eigen::Matrix3d, Eigen::RowVector3d> transform)
{

    Eigen::MatrixXd newOutput;
    newOutput.resize(vertices.rows(), vertices.cols());
    newOutput.setZero();

    // p = Rq + t
    for (size_t i=0;i<vertices.rows();i++)
    {
        Eigen::Vector3d row = vertices.row(i);
        newOutput.row(i) = (transform.first * row).transpose() + transform.second;
    }

    return newOutput;
}

// As input we give m2, m1 and the percentage of vertices to drop (subsample)
// as output we get the new M2 after icp occured
Eigen::MatrixXd ICP::ICPOptimised(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, double subsamplePercentage)
{
    Eigen::MatrixXd subsampledVertices = GetSubsample(vertices, subsamplePercentage);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> correspondences = FindCorrespondences(targetVertices, subsampledVertices);
    std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> transform = ICP::EstimateRigidTransform(correspondences.first, correspondences.second);
    return ICP::ApplyRigidTransform(vertices, transform.second);
}

// as input we give m2 and m1
// we calculate the normals and priceed with icp based on it
// as output we get the new m2
Eigen::MatrixXd ICP::ICPNormalBased(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices)
{
   Eigen::MatrixXd normals = GetNormals(targetVertices);
   std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::MatrixXd> corespond = FindCorrespondencesNormalBased(targetVertices, vertices, normals);
   std::pair<Eigen::Matrix3d, Eigen::RowVector3d> transform = ICP::EstimateRigidTransformNormalBased(corespond.first.first, corespond.second, corespond.first.second);
   return ICP::ApplyRigidTransform(vertices, transform);
}


// calculate the error metric. input is M2 and M1
// return normalised error
double ICP::GetErrorMetric(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices)
{
    double error_metric = 0.0;
    for (size_t i = 0; i < targetVertices.rows(); i++)
    {
        Eigen::RowVector3d V_i_target = targetVertices.row(i);
        Eigen::RowVector3d V_i_to_process = vertices.row(i);
        error_metric += (V_i_target - V_i_to_process).squaredNorm();
    }

    return error_metric/targetVertices.rows();
}