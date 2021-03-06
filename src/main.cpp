#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include <string> // stof, stoi

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

	createShader( // Maps x/y pos directly to UV to draw 2d image
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
	createShader(
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

int main(int argc, char* argv[]) {

	int index = 0;
	static struct option options[] = {
		{"across", required_argument, 0, 0},
		{"down", required_argument, 0, 1},
		{"timerate", required_argument, 0, 2},
		{"loss", required_argument, 0, 3},
		{"attraction", required_argument, 0, 4},
		{"gravity", required_argument, 0, 5},
		{"count", required_argument, 0, 6},
		{"speed", required_argument, 0, 7},
		{"radius", required_argument, 0, 8},
		{"circles", no_argument, 0, 9},
		{"error", no_argument, 0, 10},
		{"dt", no_argument, 0, 11},
		{0, 0, 0, 0}
	};

	while (true) {
		int arg = getopt_long(argc, argv, "", options, &index);

		if (arg == -1) {
			break;
		}

		switch (arg) {
			case 0:
				REGIONS_ACROSS = stoi(optarg);
				REGION_WIDTH = PHYSICS_WIDTH / REGIONS_ACROSS;
				break;
			
			case 1:
				REGIONS_DOWN = stoi(optarg);
				REGION_HEIGHT = PHYSICS_HEIGHT / REGIONS_DOWN;
				break;

			case 2:
				SIM_SPEED = stof(optarg);
				break;

			case 3:
				ENERGY_LOSS = stof(optarg);
				break;

			case 4:
				ATTRACTION = stof(optarg);
				break;

			case 5:
				GRAVITY = stof(optarg);
				break;

			case 6:
				PARTICLE_COUNT = stoi(optarg);
				break;

			case 7:
				PARTICLE_SPEED = stof(optarg);
				break;

			case 8:
				PARTICLE_RADIUS = stoi(optarg);
				PARTICLE_RADIUS2 = PARTICLE_RADIUS * PARTICLE_RADIUS * 4;

			case 9:
				DRAW_CIRCLES = true;
				break;

			case 10:
				SAMPLE_ERROR = true;
				break;
			
			case 11:
				PRINT_DT = true;
				break;

			case '?':
				fprintf(stderr, "Invalid option!\n");
				break;

			default:
				break;
		}
	}

	// Drawing code is as simple as possible while still being fast here.
	// All I have is a plane that fills the entire window, and a texture on that plane
	// Then I simply write individual pixels to that texture from PIXEL_BUFFER_A
	// This seems to be plenty fast for my purposes.

	GLFWwindow* window;

	if (!glfwInit()) {
		glfwTerminate();
		return -1;
	}

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Physics Simulation", NULL, NULL);
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
		srand(time(NULL));
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
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glUseProgram(shaderProgram);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glDisableVertexAttribArray(0);

		glfwSwapBuffers(window);
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