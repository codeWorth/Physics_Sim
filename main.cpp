#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <iostream>
#include <chrono>

#include "constants.h"
#include "particles.h"

using namespace std::chrono;
using namespace std;

static const GLfloat SQUARE_VERTECIES[] = {
	-1.0f, -1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f, 1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,
};

bool createShader(char const* code, GLenum shaderType, GLuint& id) {
	GLuint shaderID = glCreateShader(shaderType);
	glShaderSource(shaderID, 1, &code, NULL);
	glCompileShader(shaderID);

	GLint result;
	int logLength;
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logLength);
	if (logLength > 0) {
		char* logInfo = new char[logLength+1];
		glGetShaderInfoLog(shaderID, logLength, NULL, logInfo);
		printf("%s\n", logInfo);
		delete[] logInfo;
		return false;
	}

	id = shaderID;
	return true;
}

GLuint createProgram(GLuint const* shaders, int count) {
	GLuint programID = glCreateProgram();
	GLint result;
	int logLength;
	
	for (int i = 0; i < count; i++) {
		glAttachShader(programID, shaders[i]);

		glGetProgramiv(programID, GL_COMPILE_STATUS, &result);
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logLength);
		if (logLength > 0) {
			char* logInfo = new char[logLength+1];
			glGetShaderInfoLog(shaders[i], logLength, NULL, logInfo);
			printf("%s\n", logInfo);
			delete[] logInfo;
		}
	}
	glLinkProgram(programID);

	for (int i = 0; i < count; i++) {
		glDetachShader(programID, shaders[i]);
		glDeleteShader(shaders[i]);
	}

	return programID;
}

GLuint setupShaderProgram() {
	const int shaderCount = 2;
	GLuint shaderIds[shaderCount];

	createShader(
		"#version 330 core\n\
		layout(location = 0) in vec3 vertexPos;\
		out vec2 UV;\
		void main() {\
			gl_Position.xyz = vertexPos;\
			gl_Position.w = 1.0;\
			UV = vec2(gl_Position.x/2 + 0.5, gl_Position.y/2 + 0.5);\
		}",
		GL_VERTEX_SHADER,
		shaderIds[0]
	);
	createShader( //texture(textureSampler, UV).rgb;
		"#version 330 core\n\
		in vec2 UV;\
		out vec3 color;\
		uniform sampler2D textureSampler;\
		void main() {\
			color = texture(textureSampler, UV).rgb;\
		}", 
		GL_FRAGMENT_SHADER,
		shaderIds[1]
	);

	return createProgram(shaderIds, shaderCount);

}

int main(void) {
	GLFWwindow* window;

	if (!glfwInit()) {
		glfwTerminate();
		return -1;
	}

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Window", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	gladLoadGL();
	glfwSwapInterval(1);

	GLuint shaderProgram = setupShaderProgram();
	
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	GLuint vertexBuffer;
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(SQUARE_VERTECIES), SQUARE_VERTECIES, GL_STATIC_DRAW);

	thread PHYSICS_THREAD([]() {
		Particles particles(PARTICLE_COUNT);
		srand(105); // srand(time(NULL));
		particles.setup();
		while (true) {
			particles.tick();
		}
	});

	high_resolution_clock timer;
	const long nanoPerFrame = 16666666; // 60 fps

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	while (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0) {

		auto t0 = timer.now();
				
		// only ever read from PIXEL_BUFFER_A
		swapMutex.lock();
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, PIXEL_BUFFER_A);
		swapMutex.unlock();

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(
			0,
			3,
			GL_FLOAT,
			GL_FALSE,
			0,
			(void*)0
		);

		glUseProgram(shaderProgram);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glDisableVertexAttribArray(0);

		glfwSwapBuffers(window); /* Swap front and back buffers */
		glfwPollEvents();

		long dt = duration_cast<nanoseconds>(timer.now() - t0).count(); // maintain max of 60 fps
		dt = max(0L, nanoPerFrame - dt);
		this_thread::sleep_for(nanoseconds(dt));

	}

	delete[] PIXEL_BUFFER_A;
	delete[] PIXEL_BUFFER_B;

	PHYSICS_THREAD.detach();

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}