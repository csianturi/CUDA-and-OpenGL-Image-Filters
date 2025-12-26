#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<stb/stb_image.h>
#include<glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include<glm/gtc/matrix_transform.hpp>

#include "shaderClass.h"
#include "Texture.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"


const unsigned int width = 800;
const unsigned int height = 800;


// Vertices coordinates
GLfloat vertices[] =
{ //     COORDINATES     /        COLORS      /   TexCoord  //
	-0.5f, 0.0f,  0.5f,     0.83f, 0.70f, 0.44f,	0.0f, 0.0f,
	-0.5f, 0.0f, -0.5f,     0.83f, 0.70f, 0.44f,	5.0f, 0.0f,
	 0.5f, 0.0f, -0.5f,     0.83f, 0.70f, 0.44f,	0.0f, 0.0f,
	 0.5f, 0.0f,  0.5f,     0.83f, 0.70f, 0.44f,	5.0f, 0.0f,
	 0.0f, 0.8f,  0.0f,     0.92f, 0.86f, 0.76f,	2.5f, 5.0f
};

// Indices for vertices order
GLuint indices[] =
{
	0, 1, 2,
	0, 2, 3,
	0, 1, 4,
	1, 2, 4,
	2, 3, 4,
	3, 0, 4
};

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "OpenGL", NULL, NULL);
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
	glViewport(0, 0, width, height);
	
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
	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 8 * sizeof(float), (void*)0);
	VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	VAO1.LinkAttrib(VBO1, 2, 2, GL_FLOAT, 8 * sizeof(float), (void*)(6 * sizeof(float)));

	// Unbind all to prevent accidental modifications
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

	Texture deku("deku.png", GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
	deku.texUnit(shaderProgram, "tex0", 0);

	GLuint uniID = glGetUniformLocation(shaderProgram.ID, "scale");

	float rotation = 0.0f;
	double prevTime = glfwGetTime();

	glEnable(GL_DEPTH_TEST);



	// ----------- Render Loop ----------------
	//		1 iteration is a frame

	while (!glfwWindowShouldClose(window))
	{

		// Draws to the back buffer using the color, shaders, VAO, and actually drawing
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// Tell OpenGL which shader Program
		shaderProgram.Activate();

		double crntTime = glfwGetTime();
		if (crntTime - prevTime >= 1 / 60)
		{
			rotation += 0.01f;
			prevTime = crntTime;
		}

		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::mat4 proj = glm::mat4(1.0f);

		model = glm::rotate(model, glm::radians(rotation), glm::vec3(0.0f, 1.0f, 0.0f));
		view = glm::translate(view, glm::vec3(0.0f, -0.5f, -2.0f));
		proj = glm::perspective(glm::radians(45.0f), (float)(width / height), 0.1f, 100.0f);

		int modelLoc = glGetUniformLocation(shaderProgram.ID, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
		int viewLoc = glGetUniformLocation(shaderProgram.ID, "view");
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
		int projLoc = glGetUniformLocation(shaderProgram.ID, "proj");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));


		glUniform1f(uniID, 0.5f);
		deku.Bind();
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
		glDrawElements(GL_TRIANGLES, sizeof(indices)/sizeof(int), GL_UNSIGNED_INT, 0);
		// Swaps the back buffer to the front to display
		glfwSwapBuffers(window);

		//takes care of all glfw events
		glfwPollEvents();
	}

	// delete all the objects we've created
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	deku.Delete();
	shaderProgram.Delete();

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
