#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>


void onError(int code, const char* desc){ std::fprintf(stderr,"GLFW %d: %s\n", code, desc); }

int main(){
    glfwSetErrorCallback(onError);
    if(!glfwInit()){ std::fprintf(stderr,"GLFW init failed\n"); return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(800,600,"GL Prep",nullptr,nullptr);
    if(!win){ std::fprintf(stderr,"Window fail\n"); return -1; }
    glfwMakeContextCurrent(win);
    if(!gladLoadGL()){ std::fprintf(stderr,"GLAD fail\n"); return -1; }

    while(!glfwWindowShouldClose(win)){
        glClearColor(0.1f,0.1f,0.12f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(win);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
