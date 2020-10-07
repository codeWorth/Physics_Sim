# High Efficiency Physics Simulation in C++

This program simulates many identical particles under the following conditions:
1. They can bounce off each other, losing some amount of energy each time
2. They gravitational attract to each other (using the inverse square law)
3. They bounce off the walls of the simulation
4. They can be subjected to a universal gravitational field going downwards

The goal of this program is to allow simulation of two general situations, but there can be other situations if you mess with the parameters.
1. Simulate many "planets" in space under gravitational force
2. Simulate a fluid

In order to achieve this goal, it's important for this simulation to be as efficient as possible.
I used a variety of techniques to accomplish this, which I will list here.
In addition, there are some minor optimization details commented in the code.

## Multithreading
Currently, there are two threads: one for physics, and one for rendering. This allows the physics thread to not be dragged down by the slow rendering thread.
I'm planning to add multithreading for even within the physics thread soon.

## SIMD
I use SIMD whereever possible. SIMD refers to a set of CPU instructions which can perform operations on a vector of values simulatenously, in roughly the same amount of time as doing the operation on a single value. In my case, I'm using 32 bit floats and AVX2, which includes 256 bit AVX vectors, meaning I can operate on 8 floats at once. This leads to complications because often arrays don't evenly fit into groups of 8.

## Regions
In order to make calculations more efficient, I divide the coordinate space into regions. Currently the constants.h file defines 16x16 regions, which was found to be most efficient through trial and error. Particles are then processed by region. Specifically, there are two optimizations that make use of these regions. Firstly, collision detection only needs to check adjacent and diagonally adjacent regions for particles to collide with. This greatly reduces how many distance checks are required. Secondly, gravitational attraction uses an approximation for regions that are significantly far away. Instead of calculation attractions between each particle individually, particles treat far away regions as one big particle. Every frame, each regions' center of mass is found, and this is used instead of calculation distance to each particle individually. This is only relatively accurate for regions that are some distance away, defined by `INV_LAW_RADIUS`. To test what that value needed to be, I sample a random particle each frame, and calculate its acceleration to all particles individually. Then I compare that to the approximation found with the method I just described. Set `SAMPLE_ERROR` to see a running average of this error. In order to easily and efficiently group particles by region, I use a custom made data structure I will now describe, called `GroupedArray`

## GroupedArray
This is a custom data structure made by me. It is something like a mixture of an unordered and ordered array. It is a single fixed sized array, size equal to the number of particles. Alongside the data array, it also stored an array of indecies. These indecies represent the dividing line bewteen groups within the data array. These groups are arranged sequentially. When a particle needs to change which group it is in, the process is relatively fast. Instead of removing the particle and shifting all of the data values back by one, then insterting the particle somewhere else, GroupedArray is more clever. It simply moves the first or last value of each group to effectively shift the entire group forward or backward by one. As a result of this, order is **not** preserved within each group. However, the groups stay in the same location relative to each other.

There's also a flag `#define NDEBUG` at the top of GroupedArray.h. If commented out, performance will be lower, but each function with check that a legal operation is being executed.

# Compilation and running
This project can be compiled with cmake.

The follow are command line arguments, all are optional:

	--across:	number of regions in x direction, default 16 (int)
	--down:		number of regions in y direction, default 16 (int)
	--timerate:	scaling that time progresses at, default 1 (float)
	--loss:		percent of energy retained during collisions, default 1, (0, 1] (float)
	--attraction:gravitational attraction between particles, default 30000 (float)
	--gravity:	universal gravity downwards, default 0, try 60 (float)
	--count:	number of particles, default 5000 (int)
	--speed:	initial randomized speed of particles, default 400 (float)
	--radius:	particle radius, default 16 (int)
	--circles:	flag to draw particles as circles, rather than just points, defaults to false
	--error:	flag to print sampled acceleration error, defaults to false
	--dt:		flag to print delta time between ticks, defaults to false