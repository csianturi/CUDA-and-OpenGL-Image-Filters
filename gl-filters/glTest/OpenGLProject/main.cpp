#include <iostream>
#include <glad/glad.h>
#include<GLFW/glfw3.h>

// Vertex Shader source code
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
//Fragment Shader source code
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(0.8f, 0.3f, 0.02f, 1.0f);\n"
"}\n\0";

int main() {
	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	// Vertices coordinates
	GLfloat vertices[] =
	{
		-0.5f, -0.5f * float(sqrt(3)) / 3 , 0.0f,		// lower left corner
		0.5f, -0.5f * float(sqrt(3)) / 3 , 0.0f,		// lower right corner
		0.0f, 0.5f * float(sqrt(3)) * 2 / 3 , 0.0f		// upper corner
	};

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

	// Create Vertex Shader Obj and get reference
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Attach Vertex Shader source to the Vertex Shader Object
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	// Compile the vertex shader 
	glCompileShader(vertexShader);

	// Create Fragment shader obj and get reference
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	// Attach fragment shader source to the fragment shader object;
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	// Compile the fragment shader
	glCompileShader(fragmentShader);

	//Create Shader Program Object and get its reference
	GLuint shaderProgram = glCreateProgram();
	// Attach the Vertex and Fragment Shaders to the Shader Program
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	// Wrap up/Link all the shaders together into the shader program
	glLinkProgram(shaderProgram);

	// delete the now useless vertex and fragment shader objects
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);




	// ----------- VAO/VBO Init ----------------

	// Create reference containers for the Vertex Array Obj and Vertex Buffer Obj
	GLuint VBO, VAO;

	// Generate the VAO and VBO with only 1 obj each ( VAO FIRST!!!!)
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	// Make the VAO the current Vertex Array Obj by binding it
	glBindVertexArray(VAO);

	// Bind the VBO specifying its type 
	glBindBuffer(GL_ARRAY_BUFFER, VBO);			// make this buffer (VBO) the current vertex attribute buffer

	// Introduce the vertices into the VBO
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Configure the Vertex Attribute so that OpenGL knows how to read the VBO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);		// basically sets VAO configuration
	// Enable the Vertex Attribute so that OpenGL knows how to use it
	glEnableVertexAttribArray(0);														

	// Bind both the VAO and VBO so no accidental modifications are made
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);





	// ----------- Initial back/front buffer swap ----------------
	

	// specify color of the background
	glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
	// clean the back buffer and assign the new color to it
	glClear(GL_COLOR_BUFFER_BIT);
	// swap the back buffer with the front buffer
	glfwSwapBuffers(window);






	// ----------- Render Loop ----------------
	//		1 iteration is a frame

	while (!glfwWindowShouldClose(window))
	{

		// Draws to the back buffer using the color, shaders, VAO, and actually drawing
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		// Tell OpenGL which shader Program
		glUseProgram(shaderProgram);
		// bind the VAO so OpenGL knows how to use it
		glBindVertexArray(VAO);
		// Draw the triangle using the GL_TRIANGLES primitive
		
		// Drawing Pipeline:
		// 1. Vertex Fetch - Uses VAO to figure out how to input VBO data into the Vertex Shader
		// 2. Vertex Shader - Processes for each vertex
		// 3. Primitive Assembly - Every group of 3 vertices will become GL_TRIANGLE
		// 4. Clipping and Perspective Divide - Triangle is in clip space (3D Cube where visible coords range -1 to 1)
		//     OpenGL will transform it into Normalized Device Coords (NDC) -1<x<1 and -1<y<1
		// 5. Vieport Transform - NDC coordinates are mapped to glViewport(value);
		// 6. Rasterization - Triangle is broken down into fragments (potential pixels)
		// 7. Fragment Shader - Runs once per fragment and sets final pixel color
		// 8. Output Merger - fragment colors are written to the color buffer
		glDrawArrays(GL_TRIANGLES, 0, 3);
		// Swaps the back buffer to the front to display
		glfwSwapBuffers(window);

		//takes care of all glfw events
		glfwPollEvents();
	}

	// delete all the objects we've created
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteProgram(shaderProgram);

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
