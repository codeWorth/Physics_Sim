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

const int REGIONS_ACROSS = 16;
const int REGIONS_DOWN = 16;
const int REGION_WIDTH = PHYSICS_WIDTH / REGIONS_ACROSS;
const int REGION_HEIGHT = PHYSICS_HEIGHT / REGIONS_DOWN;

const GLfloat SIM_SPEED = 1;
const GLfloat ENERGY_LOSS = 0.97;
const GLfloat ATTRACTION = 30000;
const GLfloat GRAVITY = 0;

const int PARTICLE_COUNT = 5000;
const int PARTICLE_RADIUS = PHYSICS_SCALE * 2;
const int PARTICLE_RADIUS2 = PARTICLE_RADIUS * PARTICLE_RADIUS * 4;
const GLfloat PARTICLE_SPEED = 400;

std::mutex swapMutex;

#endif // PHYSICS_CONSTANTS
