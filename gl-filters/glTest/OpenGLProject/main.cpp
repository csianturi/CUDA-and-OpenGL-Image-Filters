#include <iostream>
#include <glad/glad.h>
#include<GLFW/glfw3.h>

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 800, "OpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "failed to create window" << std::endl;
		glfwTerminate();
		return -1;
	}
	
	//introduce window into current context
	glfwMakeContextCurrent(window);

	//load glad so it configures opengl
	gladLoadGL();

	//specify the size of the viewport of opengl in the window
	glViewport(0, 0, 800, 800);

	// specify color of the background
	glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
	// clean the back buffer and assign the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
	// swap the back buffer with the front buffer
	glfwSwapBuffers(window);

	while (!glfwWindowShouldClose(window))
	{
		//takes care of all glfw events
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
