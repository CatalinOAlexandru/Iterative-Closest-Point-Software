#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "scene.h"

int main(int argc, char *argv[]){

    srand (time(NULL));

    // Creating the viewer and adding plugins
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);
    Scene scene(viewer);

	// GUI Variables and default values
    int stepIteration = 1;
    int iteration = 200;
	double rotX = 0.0;
    double rotY = 0.0;
    double rotZ = 0.0;
    double gausSTD = 0.0; // gaussian blur STD
    double subsamplePercentage = 0.0;
    
    // New window on the screen
    menu.callback_draw_custom_window = [&]()
    {
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(450, 300), ImGuiSetCond_FirstUseEver);
        ImGui::Begin( "Tasks", nullptr, ImGuiWindowFlags_NoSavedSettings );

        if (ImGui::CollapsingHeader("Utilities", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if(ImGui::InputInt("Iteration", &iteration))
            {
                scene.SetIteration(iteration);
            }
            if (ImGui::Button("Reset Viewer / ICP", ImVec2(-1, 0)))
            {
                scene.Initialise();
                stepIteration = 1;
            }
            if (ImGui::Button("Reset Viewer (No Meshes)", ImVec2(-1, 0)))
            {
                scene.Reset();
            }
        }

        if (ImGui::CollapsingHeader("Task 1", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Start ICP", ImVec2(-1, 0))){
                scene.Point2PointICP();
            }
            // Run ICP for only 1 iteration. 2nd click will run for 2 but from the beginning and so on
            // Not very good as it resets everytime but it can act as an animation.
            if (ImGui::Button("1 Step per Click ICP", ImVec2(-1, 0))){
                scene.SetIteration(stepIteration);
                scene.Point2PointICP();
                stepIteration++;
            }
        }
        
        if (ImGui::CollapsingHeader("Task 2", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Rotate X Degrees", &rotX, 0, 0, "%.4f");
            ImGui::InputDouble("Rotate Y Degrees", &rotY, 0, 0, "%.4f");
            ImGui::InputDouble("Rotate Z Degrees", &rotZ, 0, 0, "%.4f");

            if (ImGui::Button("Rotate Mesh", ImVec2(-1, 0))){
                scene.Rotation(rotX,rotY,rotZ);
            }
        }

        if (ImGui::CollapsingHeader("Task 3", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("Noise STD", &gausSTD, 0, 0, "%.4f");

            if (ImGui::Button("Add Noise", ImVec2(-1, 0))){
                scene.Perdurbation(gausSTD);
            }
        }

        if (ImGui::CollapsingHeader("Task 4", ImGuiTreeNodeFlags_DefaultOpen))
        {

            if(ImGui::InputDouble("Subsample Percentage", &subsamplePercentage, 0, 0, "%.4f"))
            {
                scene.SetSubsample(subsamplePercentage);
            }

            if (ImGui::Button("Start Subsampled ICP", ImVec2(-1, 0))){
                scene.Point2PointICPOptimised();
            }
        }
        
        if (ImGui::CollapsingHeader("Task 5", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Load All Meshes", ImVec2(-1, 0))){
                scene.LoadAllMeshes();
            }
            
            if (ImGui::Button("Start Multi Mesh ICP", ImVec2(-1, 0))){
                scene.MultiMeshAlign();
            }
        }
        
        if (ImGui::CollapsingHeader("Task 6", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Normals Based ICP", ImVec2(-1, 0))){
                scene.Point2PlaneICP();
            }
        }
        
        ImGui::End();
    };

    // Initialise the scene
	scene.Initialise();

	// Call GUI
	viewer.launch();

}