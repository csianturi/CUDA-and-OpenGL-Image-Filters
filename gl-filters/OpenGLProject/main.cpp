#include <iostream>
#include <glad/glad.h>
#include<GLFW/glfw3.h>

#include "shaderClass.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"

// Vertices coordinates
GLfloat vertices[] =
{
	-0.5f, -0.5f * float(sqrt(3)) / 3 , 0.0f,	0.8f, 0.3f, 0.02f,	// lower left corner
	0.5f, -0.5f * float(sqrt(3)) / 3 , 0.0f,	0.8f, 0.3f, 0.02f,	// lower right corner
	0.0f, 0.5f * float(sqrt(3)) * 2 / 3 , 0.0f,	1.0f, 0.6f, 0.32f,	// upper corner
	-0.5f / 2, 0.5f * float(sqrt(3)) / 6 , 0.0f,0.9f, 0.45f, 0.17f,	// inner left
	0.5f / 2, 0.5f * float(sqrt(3)) / 6 , 0.0f,	0.9f, 0.45f, 0.17f,	// inner right
	0.0f , -0.5f * float(sqrt(3)) / 3 , 0.0f,	0.8f, 0.3f, 0.02f	// inner down

};

// indices of the vertices Ex. vertices[0],[3],[5] form lower left triangle
GLuint indices[] =
{
	0, 3, 5,	// lower left triangle
	3, 2, 4,	// lower right triangle
	5, 4, 1		// upper triangle
};

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
	
	// ----------- Shader Init ----------------
	Shader shaderProgram("default.vert", "default.frag");

	// ----------- VAO/VBO/EBO Init ----------------

	// initialize the VAO and bind it
	VAO VAO1;
	VAO1.Bind();

	// Initializes VBO/EBO and link them to verteces/indices
	VBO VBO1(vertices, sizeof(vertices));
	EBO EBO1(indices, sizeof(indices));

	// Links VBO to VAO
	// void LinkAttrib(VBO& VBO, GLuint layout, GLuint numComponents, GLenum type, GLsizeiptr stride, void* offset);
	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 6 * sizeof(float), (void*)0);
	VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 6 * sizeof(float), (void*)(3 * sizeof(float)));

	// Unbind all to prevent accidental modifications
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

	GLuint uniID = glGetUniformLocation(shaderProgram.ID, "scale");

	/*
	// Create reference containers for the Vertex Array Obj and Vertex Buffer Obj and Element Buffer Obj
	GLuint VBO, VAO, EBO;

	// Generate the VAO, VBO, and EBO with only 1 obj each ( VAO FIRST!!!!)
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Make the VAO the current Vertex Array Obj by binding it
	glBindVertexArray(VAO); //ACTIVATES

	// Bind the VBO specifying its type 
	glBindBuffer(GL_ARRAY_BUFFER, VBO);				// make this buffer (VBO) the current vertex attribute buffer
		
	// Introduce the vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Bind the EBO specifying its type and allocate and assign its memory
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Configure the Vertex Attribute so that OpenGL knows how to read the VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);		// basically sets VAO configuration
	// Enable the Vertex Attribute so that OpenGL knows how to use it
	glEnableVertexAttribArray(0);														

	// Bind both the VAO and VBO so no accidental modifications are made
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	*/



	// ----------- Render Loop ----------------
	//		1 iteration is a frame

	while (!glfwWindowShouldClose(window))
	{

		// Draws to the back buffer using the color, shaders, VAO, and actually drawing
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		// Tell OpenGL which shader Program
		shaderProgram.Activate();
		glUniform1f(uniID, 0.5f);
		// bind the VAO so OpenGL knows how to use it
		VAO1.Bind();
		// Draw the triangle using the GL_TRIANGLES primitive
		
		// Drawing Pipeline:
		// 1. Vertex Fetch - Uses VAO to figure out how to input VBO data into the Vertex Shader
		// 2. Vertex Shader - Processes for each vertex
		// 3. Primitive Assembly - Every group of 3 vertices will become GL_TRIANGLE
		// 4. Clipping and Perspective Divide - Triangle is in clip space (3D Cube where visible coords range -1 to 1)
		//     OpenGL will transform it into Normalized Device Coords (NDC) -1<x<1 and -1<y<1
		// 5. Viewport Transform - NDC coordinates are mapped to glViewport(value);
		// 6. Rasterization - Triangle is broken down into fragments (potential pixels)
		// 7. Fragment Shader - Runs once per fragment and sets final pixel color
		// 8. Output Merger - fragment colors are written to the color buffer
		
		//glDrawArrays(GL_TRIANGLES, 0, 3);
		
		// uses the 9 indices and groups them into 3 triangles based on the indices
		glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
		// Swaps the back buffer to the front to display
		glfwSwapBuffers(window);

		//takes care of all glfw events
		glfwPollEvents();
	}

	// delete all the objects we've created
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	shaderProgram.Delete();

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
