class Scene
{
    public:
        Scene(igl::opengl::glfw::Viewer& refViewer);
        ~Scene();
        
        // Task 1.1
        void Point2PointICP();
        // Task 2
        void Rotation(double x, double y, double z);
        // Task 3
        void Perdurbation(double std);
        // Task 4
        void Point2PointICPOptimised();
        // Task 5
        void LoadAllMeshes();
        void MultiMeshAlign();
        // Task 6
        void Point2PlaneICP();
        // Other
        void Initialise();
        void Reset();
        void Visualise(int i);
        void SetIteration(int i);
        void SetSubsample(double amount);
        
    private:
        
        igl::opengl::glfw::Viewer& viewer;
        
        Eigen::MatrixXd V1, V2, V3, V4, V5, V6;
        Eigen::MatrixXi F1, F2, F3, F4, F5, F6;
        
        int numIters;
        double subsamplePercentage;

        struct RenderingData;
        std::vector<RenderingData> renderingData;
};
