#if !defined(PHYSICS_CONSTANTS)
#define PHYSICS_CONSTANTS

#include <glad/glad.h>
#include <mutex>

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const int PIXEL_COUNT = WINDOW_WIDTH * WINDOW_HEIGHT;
char* PIXEL_BUFFER_A = new char[PIXEL_COUNT];
char* PIXEL_BUFFER_B = new char[PIXEL_COUNT];

const int PHYSICS_SCALE_POWER = 3;
const int PHYSICS_SCALE = 1 << PHYSICS_SCALE_POWER;
const int PHYSICS_WIDTH = WINDOW_WIDTH * PHYSICS_SCALE;
const int PHYSICS_HEIGHT = WINDOW_HEIGHT * PHYSICS_SCALE;

int REGIONS_ACROSS = 16;
int REGIONS_DOWN = 16;
int REGION_WIDTH = PHYSICS_WIDTH / REGIONS_ACROSS;
int REGION_HEIGHT = PHYSICS_HEIGHT / REGIONS_DOWN;

GLfloat SIM_SPEED = 1;
GLfloat ENERGY_LOSS = 0.97;
GLfloat ATTRACTION = 30000;
GLfloat GRAVITY = 0;

int PARTICLE_COUNT = 5000;
int PARTICLE_RADIUS = PHYSICS_SCALE * 2;
int PARTICLE_RADIUS2 = PARTICLE_RADIUS * PARTICLE_RADIUS * 4;
GLfloat PARTICLE_SPEED = 400;

std::mutex swapMutex;


bool DRAW_CIRCLES = false;	// draw the full circle for each particle, or just a single pixel
bool SAMPLE_ERROR = false;	// print acceleration error sampled from random particles

#endif // PHYSICS_CONSTANTS
