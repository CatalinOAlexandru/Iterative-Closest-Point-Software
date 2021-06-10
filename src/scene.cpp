#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include "scene.h"
#include "icp.h"
#define PATH "../data/"

struct Scene::RenderingData
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;
};

Scene::Scene(igl::opengl::glfw::Viewer& refViewer):viewer(refViewer)
{
    numIters = 250;
    subsamplePercentage = 0;
}

Scene::~Scene(){}

void Scene::Reset(){

    renderingData.clear();
    Visualise(0);
}

void Scene::Initialise()
{
    
    renderingData.clear();

    igl::readOFF(PATH "bun000_v2.off", V1, F1);
    igl::readOFF(PATH "bun045_v2.off", V2, F2);

    Eigen::MatrixXd V(V1.rows()+V2.rows(), V1.cols());
    V << V1,V2;
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1,(F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C <<
    Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),
    Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});
    
    Visualise(renderingData.size());

}

void Scene::Point2PointICP()
{
    
    renderingData.clear();

    Eigen::MatrixXd Vx = V2;

    int totalNumIters = numIters;

    clock_t timer = std::clock();

    double error_metric = 1000;

    for (size_t i=0; i<numIters;i++)
    {
        // Basic ICP algorithm
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> correspondences = ICP::FindCorrespondences(V1, Vx);
        std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> transform_info = ICP::EstimateRigidTransform(correspondences.first, correspondences.second);
        Vx = ICP::ApplyRigidTransform(Vx, transform_info.second);
    }

    double timeSeconds = (clock() - timer) / (double) CLOCKS_PER_SEC;
    std::cout << "ICP Basic takes " + std::to_string(timeSeconds) + "s to complete " + std::to_string(totalNumIters) + " iteration(s)" << std::endl;

    // Generate data and store them for display
    Eigen::MatrixXd V(V1.rows()+Vx.rows(), V1.cols());
    V << V1,Vx;
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1,(F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C << Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});

    Visualise(renderingData.size());
}

void Scene::Rotation(double x, double y, double z)
{
    
    renderingData.clear();
    
    // Load M1
    igl::readOFF(PATH "bun000_v2.off", V1, F1);
    
    // M2 = R(M1), vertex positions are changed while the face relation remains
    V2 = ICP::Rotate(V1, x, y, z);
    F2 = F1;
    
    // Display meshes
    Eigen::MatrixXd V(V1.rows()+V2.rows(), V1.cols());
    V << V1,V2;

    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1, (F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C <<
    Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),
    Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});
    
    Visualise(renderingData.size());

}

void Scene::Perdurbation(double std)
{

    renderingData.clear();

    // Load M1
    igl::readOFF(PATH "bun000_v2.off", V1, F1);
    igl::readOFF(PATH "bun045_v2.off", V2, F2);

    // M2' = M2
    V2 = ICP::AddNoise(V2, std);

    // Display meshes
    Eigen::MatrixXd V(V1.rows()+V2.rows(), V1.cols());
    V << V1,V2;

    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1, (F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C << Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1), 
         Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});

    Visualise(renderingData.size());

}

void Scene::Point2PointICPOptimised()
{

    renderingData.clear();

    Eigen::MatrixXd Vx = V2;

    double last_distance = ICP::GetErrorMetric(V1, Vx);
    int totalNumIters = numIters;
    double error_metric = 1000;

    clock_t timer = std::clock();

    // Use the subsample to perform ICP algorithm
    for (size_t i=0; i<numIters;i++){
        Eigen::MatrixXd V_subsampled = ICP::GetSubsample(Vx, subsamplePercentage);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> correspondences = ICP::FindCorrespondences(V1, V_subsampled);
        std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> transform_info = ICP::EstimateRigidTransform(correspondences.first, correspondences.second);
        Vx =  ICP::ApplyRigidTransform(Vx, transform_info.second);
    }

    double timeSeconds = (clock() - timer) / (double) CLOCKS_PER_SEC;
    std::cout << "ICP Optimised takes " + std::to_string(timeSeconds) + "s to complete " + std::to_string(totalNumIters) + " iteration(s)" << std::endl;

    // Generate data and store them for display
    Eigen::MatrixXd V(V1.rows()+Vx.rows(), V1.cols());
    V << V1,Vx;
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1,(F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C <<
      Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),
      Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});

    Visualise(renderingData.size());
}

void Scene::LoadAllMeshes()
{
    
    renderingData.clear();

    // Method 1 as discribed int he report
    // igl::readOFF(PATH "bun000_v2.off", V1, F1);
    // igl::readOFF(PATH "bun045_v2.off", V2, F2);
    // igl::readOFF(PATH "bun090_v2.off", V3, F3);
    // igl::readOFF(PATH "bun180_v2.off", V4, F4);
    // igl::readOFF(PATH "bun270_v2.off", V5, F5);
    // igl::readOFF(PATH "bun315_v2.off", V6, F6);

    // Method 2 as discribed in the report
    igl::readOFF(PATH "bun315_v2.off", V1, F1);
    igl::readOFF(PATH "bun270_v2.off", V2, F2);
    igl::readOFF(PATH "bun000_v2.off", V3, F3);
    igl::readOFF(PATH "bun045_v2.off", V4, F4);
    igl::readOFF(PATH "bun090_v2.off", V5, F5);
    
    // Display meshes
    Eigen::MatrixXd V(V1.rows()+V2.rows()+V3.rows()+V4.rows()+V5.rows()+V6.rows(), V1.cols());
    V << V1,V2,V3,V4,V5,V6;
    
    Eigen::MatrixXi F(F1.rows()+F2.rows()+F3.rows()+F4.rows()+F5.rows()+F6.rows(), F1.cols());
    F <<F1,(F2.array()+V1.rows()), (F3.array()+V2.rows()+V1.rows()), (F4.array()+V3.rows()+V2.rows()+V1.rows()), (F5.array()+V4.rows()+V3.rows()+V2.rows()+V1.rows());
    
    Eigen::MatrixXd C(F.rows(),3);
    C <<
    Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),
    Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1),
    Eigen::RowVector3d(0.0,1.0,0.0).replicate(F3.rows(),1),
    Eigen::RowVector3d(1.0,0.0,1.0).replicate(F4.rows(),1),
    Eigen::RowVector3d(0.5,0.0,0.5).replicate(F5.rows(),1);
    // Used for method 1
    // Eigen::RowVector3d(0.2,0.8,0.5).replicate(F6.rows(),1);
    
    renderingData.push_back(RenderingData{V,F,C});
    
    Visualise(renderingData.size());
}

void Scene::MultiMeshAlign()
{

    renderingData.clear();

    Eigen::MatrixXd V1r = V1;
    Eigen::MatrixXi F1r = F1;

    Eigen::MatrixXd V2r = V2;
    Eigen::MatrixXi F2r = F2;

    Eigen::MatrixXd V3r = V3;
    Eigen::MatrixXi F3r = F3;

    Eigen::MatrixXd V4r = V4;
    Eigen::MatrixXi F4r = F4;

    Eigen::MatrixXd V5r = V5;
    Eigen::MatrixXi F5r = F5;

    // Used for method 1
    // Eigen::MatrixXd V6r = V6;
    // Eigen::MatrixXi F6r = F6;

    clock_t timer = std::clock();

    for (size_t i=0; i<numIters;i++) 
    {
        V2r = ICP::ICPOptimised(V1, V2r, 95);
    }

    Eigen::MatrixXd V12(V1.rows()+V2r.rows(), V1.cols());
    V12<<V1, V2r;
    Eigen::MatrixXi F12(F1.rows()+F2r.rows(), F1.cols());
    F12<<F1, (F2r.array()+V1.rows());

    for (size_t i=0; i<numIters;i++) 
    {
        V3r = ICP::ICPOptimised(V12, V3r, 95);
    }

    Eigen::MatrixXd V123(V12.rows()+V3r.rows(), V1.cols());
    V123<<V12, V3r;
    Eigen::MatrixXi F123(F12.rows()+F3r.rows(), F1.cols());
    F123<<F12, (F3r.array()+V12.rows());

    for (size_t i=0; i<numIters;i++) 
    {
        V4r = ICP::ICPOptimised(V123, V4r, 95);
    }

    Eigen::MatrixXd V1234(V123.rows()+V4r.rows(), V1.cols());
    V1234<<V123, V4r;
    Eigen::MatrixXi F1234(F123.rows()+F4r.rows(), F1.cols());
    F1234<<F123,(F4r.array()+V123.rows());

    for (size_t i=0; i<numIters;i++) 
    {
        V5r = ICP::ICPOptimised(V1234, V5r, 95);
    }

    Eigen::MatrixXd V12345(V1234.rows()+V5.rows(), V1.cols());
    V12345<<V1234, V5r;
    Eigen::MatrixXi F12345(F1234.rows()+F5r.rows(), F1.cols());
    F12345<<F1234,(F5r.array()+V1234.rows());

    // Used for Method 1 only
    // for (size_t i=0; i<numIters;i++) 
    // {
    //     V6r = ICP::ICPOptimised(V12345, V6r, 95);
    // }

    // Eigen::MatrixXd V123456(V12345.rows()+V6.rows(), V1.cols());
    // V123456<<V12345, V6r;
    // Eigen::MatrixXi F123456(F12345.rows()+F6r.rows(), F1.cols());
    // F123456<<F12345,(F6r.array()+V12345.rows());

    Eigen::MatrixXd C(F12345.rows(),3);
    // Used for method 1 only
    // Eigen::MatrixXd C(F123456.rows(),3);
    C<<
    Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1),
    Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2r.rows(),1),
    Eigen::RowVector3d(0.0,1.0,0.0).replicate(F3r.rows(),1),
    Eigen::RowVector3d(1.0,0.0,1.0).replicate(F4r.rows(),1),
    Eigen::RowVector3d(0.5,0.0,0.5).replicate(F5r.rows(),1);
    // Used for method 1
    // Eigen::RowVector3d(0.2,0.8,0.5).replicate(F6r.rows(),1);

    // Used for method 1
    renderingData.push_back(RenderingData{V12345,F12345,C});
    // renderingData.push_back(RenderingData{V123456,F123456,C});

    double timeSeconds = (clock() - timer) / (double) CLOCKS_PER_SEC;
    std::cout << "Multi-Mesh took " + std::to_string(timeSeconds) + " seconds over " + std::to_string(numIters) + " iters" << std::endl;



    Visualise(renderingData.size());
}

void Scene::Point2PlaneICP()
{
    renderingData.clear();

    Eigen::MatrixXd Vx = V2;

    clock_t timer = std::clock();

    // Use the subsample to perform ICP algorithm
    for (size_t i=0; i<numIters;i++)
    {
        Eigen::MatrixXd N = ICP::GetNormals(V1);
        std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::MatrixXd> correspondences = ICP::FindCorrespondencesNormalBased(V1, Vx, N);
        std::pair<Eigen::Matrix3d, Eigen::RowVector3d> transform = ICP::EstimateRigidTransformNormalBased(correspondences.first.first, correspondences.second, correspondences.first.second);
        Vx = ICP::ApplyRigidTransform(Vx, transform);
    }

    double timeSeconds = (clock() - timer) / (double) CLOCKS_PER_SEC;
    std::cout << "Point-to-Plane took " + std::to_string(timeSeconds) + " seconds to complete " + std::to_string(numIters) + " iteration(s)" << std::endl;

    // Generate data and store them for display
    Eigen::MatrixXd V(V1.rows()+Vx.rows(), V1.cols());
    V << V1,Vx;
    Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
    F << F1,(F2.array()+V1.rows());
    Eigen::MatrixXd C(F.rows(),3);
    C << Eigen::RowVector3d(1.0,0.5,0.0).replicate(F1.rows(),1), Eigen::RowVector3d(0.0,1.0,1.0).replicate(F2.rows(),1);

    renderingData.push_back(RenderingData{V,F,C});

    Visualise(renderingData.size());
}

void Scene::SetIteration(int i)
{
    if (i < 1){ numIters = 1;}
    else{ numIters = i;}
}

void Scene::Visualise(int i)
{
    if (i > 0 && i <= renderingData.size())
    {
        viewer.data().clear();
        viewer.data().set_mesh(renderingData[i-1].V, renderingData[i-1].F);
        viewer.data().set_colors(renderingData[i-1].C);
        viewer.data().set_face_based(true);
    }
    else{ viewer.data().clear();}
}

void Scene::SetSubsample(double amount)
{
    subsamplePercentage = amount;
    if (amount < 0) subsamplePercentage = 0;
    if (amount >= 100) subsamplePercentage = 99;
}
