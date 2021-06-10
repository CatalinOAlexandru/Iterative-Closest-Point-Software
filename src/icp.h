
namespace ICP{

    Eigen::MatrixXd GetSubsample(Eigen::MatrixXd vertices, double subsamplePercentage);

    Eigen::MatrixXd GetNormals(Eigen::MatrixXd targetVertices);

    std::pair<Eigen::MatrixXi, Eigen::MatrixXi> FindNonOverlappingFaces(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, Eigen::MatrixXi Ftoprocess);

    Eigen::MatrixXd Rotate(Eigen::MatrixXd vertices, double x, double y, double z);
    
    Eigen::MatrixXd AddNoise(Eigen::MatrixXd vertices, double std);

    Eigen::MatrixXd FindBestStartRotation(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> FindCorrespondences(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices);

    std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, Eigen::MatrixXd> FindCorrespondencesNormalBased(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, Eigen::MatrixXd Ntarget);

    std::pair<double, std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> EstimateRigidTransform(Eigen::MatrixXd matchedVertices, Eigen::MatrixXd vertices);
    
    std::pair<Eigen::Matrix3d, Eigen::RowVector3d> EstimateRigidTransformNormalBased(Eigen::MatrixXd matchedVertices, Eigen::MatrixXd vertices, Eigen::MatrixXd NtoProcess);

    Eigen::MatrixXd ApplyRigidTransform(Eigen::MatrixXd vertices, std::pair<Eigen::Matrix3d, Eigen::RowVector3d> transform);

    Eigen::MatrixXd ICPOptimised(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices, double subsamplePercentage);

    Eigen::MatrixXd ICPNormalBased(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices);
    
    double GetErrorMetric(Eigen::MatrixXd targetVertices, Eigen::MatrixXd vertices);
}

