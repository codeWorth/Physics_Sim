#include <glad/glad.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <immintrin.h>

#include "constants.h"
#include "GroupedArray.h"

#define DRAW_CIRCLES false
#define INV_LAW_RADIUS 2
#define SAMPLE_ERROR false

class Particles {
public:
	GroupedArray<GLfloat> x;
	GroupedArray<GLfloat> y;
	GroupedArray<GLfloat> vx;
	GroupedArray<GLfloat> vy;
	GroupedArray<GLfloat> ax;
	GroupedArray<GLfloat> ay;

	Particles(int count);
	~Particles();

	void setup();
	void tick();

private:

	std::chrono::steady_clock timer;
	std::chrono::steady_clock::time_point lastTime;
	bool hasTime;

	long tickCount;
	double dtTotal;

	void updateVelocity(GLfloat dt);
	void updatePosition(GLfloat dt);
	void wallBounce();
	void draw() const;

	void bounce();
	void attract();

	void bounceRegions(int regionA, int bStart, int bEnd);
	void bounceRegion(int region);
	void bounceParticles(int i_, int j_, GLfloat& dx, GLfloat& dy, GLfloat& r2);

	void attractRegions(int region, int start, int end);
	void attractRegion(int region);
	void attractParticles(int i_, int j_);
	void attractParticlesBoth(int i_, int j_);

	void findCoMs(int region);
	float* coMx;
	float* coMy;

	float errAv[512];
	long index = 0;

	void updateRegions();
	int particleRegion(GLfloat x, GLfloat y) const;
	int regionIndex(int i, int j) const;
	float rsqrt_fast(float x) const;

	__m256 PARTICLE_RADIUS2_256;
	__m256 PARTICLE_RADIUS_256;
	__m256 ATTRACTION_256;
	__m256 ONES_256;
	__m256 ONE_HALFS_256;
	__m256 NEGATIVE_ONE_HALFS_256;
	__m256 ZEROSF_256;
	__m256 FLOAT_ABS_256;
	__m256 ENERGY_LOSS_256;
	__m256i ROTATE_RIGHT_256;
	__m256i WINDOW_WIDTH_256;
	__m256i ZEROS_256;

};

Particles::Particles(int count) :
	x(count, REGIONS_ACROSS*REGIONS_DOWN),
	y(count, REGIONS_ACROSS*REGIONS_DOWN),
	vx(count, REGIONS_ACROSS*REGIONS_DOWN),
	vy(count, REGIONS_ACROSS*REGIONS_DOWN),
	ax(count, REGIONS_ACROSS*REGIONS_DOWN),
	ay(count, REGIONS_ACROSS*REGIONS_DOWN) 
{

	coMx = new float[REGIONS_ACROSS*REGIONS_DOWN];
	coMy = new float[REGIONS_ACROSS*REGIONS_DOWN];

	hasTime = false;
	tickCount = 0;
	dtTotal = 0;

	PARTICLE_RADIUS2_256 = _mm256_set1_ps(PARTICLE_RADIUS2);
	PARTICLE_RADIUS_256 = _mm256_set1_ps(PARTICLE_RADIUS);
	ATTRACTION_256 = _mm256_set1_ps(ATTRACTION); 
	ZEROSF_256 = _mm256_set1_ps(0); 
	ONES_256 = _mm256_set1_ps(1);
	ONE_HALFS_256 = _mm256_set1_ps(0.5f);
	NEGATIVE_ONE_HALFS_256 = _mm256_set1_ps(-0.5f);
	ENERGY_LOSS_256 = _mm256_set1_ps(ENERGY_LOSS);
	
	WINDOW_WIDTH_256 = _mm256_set1_epi32(WINDOW_WIDTH);
	ZEROS_256 = _mm256_set1_epi32(0);

	int32_t a[8] = {7, 0, 1, 2, 3, 4, 5, 6};
	ROTATE_RIGHT_256 = _mm256_loadu_si256((__m256i*) &a);

	__m256i tmp = _mm256_set1_epi32(0x7FFFFFFF);
	FLOAT_ABS_256 = *(__m256*)&tmp;

}

Particles::~Particles() {
	delete[] coMx;
	delete[] coMy;
}

void Particles::setup() {
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		GLfloat x0 = (GLfloat)rand() / RAND_MAX * PHYSICS_WIDTH;
		GLfloat y0 = (GLfloat)rand() / RAND_MAX * PHYSICS_HEIGHT;
		int groupIndex = particleRegion(x0, y0);

		x.add(x0, groupIndex);
		y.add(y0, groupIndex);
		vx.add(((GLfloat)rand() / RAND_MAX - 0.5) * 2 * PARTICLE_SPEED, groupIndex);
		vy.add(((GLfloat)rand() / RAND_MAX - 0.5) * 2 * PARTICLE_SPEED, groupIndex);
		ax.add(0, groupIndex);
		ay.add(0, groupIndex);
	}
}

void Particles::tick() {
	if (!hasTime) {
		lastTime = timer.now();
		hasTime = true;
		return;
	}

	GLfloat dt = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(timer.now() - lastTime).count() / 1000000000.0f;
	lastTime = timer.now();

	dtTotal += dt;
	tickCount++;

	if (tickCount % 512 == 0) {
		printf("%f\n", (dtTotal / (double)tickCount));
	}

	
	this->updateRegions();
	this->updatePosition(dt);
	for (int i = 0; i < REGIONS_ACROSS*REGIONS_DOWN; i++) {
		findCoMs(i);
	}
	auto start = timer.now();
	this->attract();
	long t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(timer.now() - start).count();
	this->updateVelocity(dt);
	start = timer.now();
	this->bounce();
	long t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(timer.now() - start).count();
	this->wallBounce();
	this->draw();

	if (tickCount % 128 == 0) {
		printf("attract = %lu, \t bounce = %lu\n", t1, t2);
	}

	if (tickCount % 128 == 0 && SAMPLE_ERROR) {
		GLfloat errTot = 0;
		for (int i = 0; i < 512; i++) {
			errTot += errAv[i];
		}
		errTot /= 5.12f;
		printf("\terr: %f\n", errTot);
	}

}

void Particles::updateVelocity(GLfloat dt) {
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		this->vx[i] += this->ax[i] * dt;
		this->vy[i] += this->ay[i] * dt;
		this->ax[i] = 0;
		this->ay[i] = -GRAVITY;
	}
}

void Particles::updatePosition(GLfloat dt) {
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		this->x[i] += this->vx[i] * dt;
		this->y[i] += this->vy[i] * dt;
	}
}

void Particles::updateRegions() {
	for (int region = 0; region < x.groupsCount(); region++) {
		if (x.groupSize(region) > 0) {
			int end = x.groupStart(region) + x.groupSize(region);
			for (int i = x.groupStart(region); i < end; i++) {
				int correctRegion = particleRegion(x[i], y[i]);
				float X = x[i];
				float Y = y[i];
				if (correctRegion < region) {
					x.transferBackward(i, region, correctRegion);
					y.transferBackward(i, region, correctRegion);
					vx.transferBackward(i, region, correctRegion);
					vy.transferBackward(i, region, correctRegion);
					ax.transferBackward(i, region, correctRegion);
					ay.transferBackward(i, region, correctRegion);
				} else if (correctRegion > region) {
					x.transferForward(i, region, correctRegion);
					y.transferForward(i, region, correctRegion);
					vx.transferForward(i, region, correctRegion);
					vy.transferForward(i, region, correctRegion);
					ax.transferForward(i, region, correctRegion);
					ay.transferForward(i, region, correctRegion);

					// when transfering forward, the current item is removed, and an item after it replaces it
					// therefore, we need to process the current index again
					// also, the end index goes back by 1
					i--;
					end--;
				}
			}
		}
	}
}

void Particles::bounce() {

	for (int i = 0; i < REGIONS_ACROSS; i++) {
		for (int j = 0; j < REGIONS_DOWN; j++) {
			int regionA = regionIndex(i, j);
			bounceRegion(regionA);

			if (i+1 < REGIONS_ACROSS) {
				int regionB = regionIndex(i+1, j);
				bounceRegions(regionA, x.groupStart(regionB), x.groupStart(regionB+1));
			}

			int j2 = j+1;
			if (j2 < REGIONS_DOWN) {
				int i2Lower = std::max(0, i-1);
				int i2Upper = std::min(REGIONS_ACROSS, i+2);
				bounceRegions(
					regionA, 
					x.groupStart(regionIndex(i2Lower, j2)), 
					x.groupStart(regionIndex(i2Upper, j2))
				);
			}
		}
	}

}

void Particles::attract() {

	int sampleI = rand() % REGIONS_ACROSS;
	int sampleJ = rand() % REGIONS_DOWN;
	float ax_ = 0;
	float ay_ = 0;

	for (int i = 0; i < REGIONS_ACROSS; i++) {
		for (int j = 0; j < REGIONS_DOWN; j++) {
			int regionA = regionIndex(i, j);
			if (x.groupSize(regionA) == 0) {
				continue;
			}
			int start = x.groupStart(regionA);

			if (SAMPLE_ERROR && i == sampleI && j == sampleJ) {
				for (int k = 0; k < PARTICLE_COUNT; k++) {
					if (k == start) {
						continue;
					}
					GLfloat dx = x[k] - x[start];
					GLfloat dy = y[k] - y[start];
					GLfloat invR = rsqrt_fast(dx*dx + dy*dy);
					GLfloat f = ATTRACTION * invR*invR*invR;
					ax_ += f * dx;
					ay_ += f * dy;
				}
			}

			int i2Lower = std::max(0, i-INV_LAW_RADIUS);
			int i2Upper = std::min(REGIONS_ACROSS, i+INV_LAW_RADIUS+1);
			int j2Lower = std::max(0, j-INV_LAW_RADIUS);
			int j2Upper = std::min(REGIONS_DOWN, j+INV_LAW_RADIUS+1);

			// do close attractions
			for (int j2 = j2Lower; j2 < j; j2++) { // rows within close region above j
				int dStart = x.groupStart(regionIndex(i2Lower, j2));
				int dEnd = x.groupStart(regionIndex(i2Upper, j2));
				attractRegions(regionA, dStart, dEnd);
			}
			for (int j2 = j+1; j2 < j2Upper; j2++) { // rows within close region below j
				int dStart = x.groupStart(regionIndex(i2Lower, j2));
				int dEnd = x.groupStart(regionIndex(i2Upper, j2));
				attractRegions(regionA, dStart, dEnd);
			}
			
			// row j within close region to the left of i
			int dStart = x.groupStart(regionIndex(i2Lower, j));
			int dEnd = x.groupStart(regionIndex(i, j));
			attractRegions(regionA, dStart, dEnd);

			// row j within close region to the right of i
			dStart = x.groupStart(regionIndex(i+1, j));
			dEnd = x.groupStart(regionIndex(i2Upper, j));
			attractRegions(regionA, dStart, dEnd);

			attractRegion(regionA);


			// do center of mass attraction for remaining sections
			int size = x.groupSize(regionA);
			int groupedSize = (size / 8) * 8;

			for (int k = 0; k < groupedSize; k += 8) {
				__m256 xs = _mm256_loadu_ps(x.data + start + k);
				__m256 ys = _mm256_loadu_ps(y.data + start + k);
				__m256 axs = _mm256_loadu_ps(ax.data + start + k);
				__m256 ays = _mm256_loadu_ps(ay.data + start + k);

				for (int i2 = 0; i2 < REGIONS_ACROSS; i2++) {
					for (int j2 = 0; j2 < REGIONS_DOWN; j2++) {
						int region = regionIndex(i2, j2);
						float regionSize = x.groupSize(region);
						if ((i2 >= i2Lower && i2 < i2Upper && j2 >= j2Lower && j2 < j2Upper) || regionSize == 0) {
							continue;
						}

						__m256 F_256 = _mm256_set1_ps(ATTRACTION * regionSize);
						__m256 CENTER_X_256 = _mm256_set1_ps(coMx[region]);
						__m256 CENTER_Y_256 = _mm256_set1_ps(coMy[region]);

						__m256 dx = _mm256_sub_ps(CENTER_X_256, xs);
						__m256 dy = _mm256_sub_ps(CENTER_Y_256, ys);
						__m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)); // dx*dx + dy*dy
						__m256 invR = _mm256_rsqrt_ps(r2);
						__m256 f = _mm256_mul_ps(F_256, _mm256_mul_ps(invR, _mm256_mul_ps(invR, invR))); // F / invR^3

						axs = _mm256_fmadd_ps(f, dx, axs);
						ays = _mm256_fmadd_ps(f, dy, ays);
					}
				}

				_mm256_storeu_ps(ax.data + start + k, axs);
				_mm256_storeu_ps(ay.data + start + k, ays);	
			
			}
			for (int k = groupedSize; k < size; k++) {
				for (int i2 = 0; i2 < REGIONS_ACROSS; i2++) {
					for (int j2 = 0; j2 < REGIONS_DOWN; j2++) {
						int region = regionIndex(i2, j2);
						float regionSize = x.groupSize(region);
						if ((i2 >= i2Lower && i2 < i2Upper && j2 >= j2Lower && j2 < j2Upper) || regionSize == 0) {
							continue;
						}

						GLfloat dx = coMx[region] - x[start + k];
						GLfloat dy = coMy[region] - y[start + k];
						GLfloat invR = rsqrt_fast(dx*dx + dy*dy);
						GLfloat f = ATTRACTION * regionSize * invR*invR*invR;

						ax[start + k] += f * dx;
						ay[start + k] += f * dy;
					}
				}
			}

			if (SAMPLE_ERROR && i == sampleI && j == sampleJ) {
				GLfloat errX = (ax[start] - ax_) / ax_;
				GLfloat errY = (ay[start] - ay_) / ay_;
				errAv[index % 512] = std::sqrt(errX*errX + errY*errY);
				index++;
			}

		}
	}

}

void Particles::bounceParticles(int i_, int j_, GLfloat& dx, GLfloat& dy, GLfloat& r2) {
	GLfloat invR = rsqrt_fast(r2);
	dx *= invR;
	dy *= invR;

	GLfloat midX = (x[i_] + x[j_]) / 2;
	GLfloat midY = (y[i_] + y[j_]) / 2;
	x[i_] = midX - dx*PARTICLE_RADIUS;
	y[i_] = midY - dy*PARTICLE_RADIUS;
	x[j_] = midX + dx*PARTICLE_RADIUS;
	y[j_] = midY + dy*PARTICLE_RADIUS;

	GLfloat avAlong = vx[i_]*dx + vy[i_]*dy;
	GLfloat bvAlong = vx[j_]*dx + vy[j_]*dy;
	GLfloat avxAlong = avAlong * dx;
	GLfloat avyAlong = avAlong * dy;
	GLfloat bvxAlong = bvAlong * dx;
	GLfloat bvyAlong = bvAlong * dy;
	vx[i_] += (bvxAlong - avxAlong) * ENERGY_LOSS;
	vy[i_] += (bvyAlong - avyAlong) * ENERGY_LOSS;
	vx[j_] += (avxAlong - bvxAlong) * ENERGY_LOSS;
	vy[j_] += (avyAlong - bvyAlong) * ENERGY_LOSS;
}

void Particles::attractParticles(int i_, int j_) {
	GLfloat dx = x[j_] - x[i_];
	GLfloat dy = y[j_] - y[i_];
	GLfloat invR = rsqrt_fast(dx*dx + dy*dy);
	GLfloat f = ATTRACTION * invR*invR*invR;
	ax[i_] += dx * f;
	ay[i_] += dy * f;
}

void Particles::attractParticlesBoth(int i_, int j_) {
	GLfloat dx = x[j_] - x[i_];
	GLfloat dy = y[j_] - y[i_];
	GLfloat invR = rsqrt_fast(dx*dx + dy*dy);
	GLfloat f = ATTRACTION * invR*invR*invR;
	ax[i_] += dx * f;
	ay[i_] += dy * f;
	ax[j_] -= dx * f;
	ay[j_] -= dy * f;
}

void Particles::findCoMs(int region) {
	int size = x.groupSize(region);
	if (size == 0) {
		coMx[region] = 0;
		coMy[region] = 0;
		return;
	}

	int start = x.groupStart(region);
	int groupedSize = (size / 8) * 8;

	__m256 sumX = _mm256_set1_ps(0);
	__m256 sumY = _mm256_set1_ps(0);
	for (int i = 0; i < groupedSize; i += 8) {
		sumX = _mm256_add_ps(sumX, _mm256_loadu_ps(x.data + start + i));
		sumY = _mm256_add_ps(sumY, _mm256_loadu_ps(y.data + start + i));
	}

	float totalX = 0;
	float totalY = 0;
	for (int i = 0; i < 8; i++) {
		totalX += ((float*)&sumX)[i];
		totalY += ((float*)&sumY)[i];
	}

	for (int i = groupedSize; i < size; i++) {
		totalX += x[start + i];
		totalY += y[start + i];
	}

	coMx[region] = totalX / (float)size;
	coMy[region] = totalY / (float)size;

}

void Particles::attractRegions(int region, int start, int end) {
	if (start == end) {
		return;
	}
	int regionStart = x.groupStart(region);
	int regionSize = x.groupSize(region);
	int bSize = end - start;
	int simdSize = (regionSize / 8) * 8;

	for (int i = 0; i < simdSize; i += 8) {
		for (int i_ = 0; i_ < 8; i_++) {
			for (int j = 0; j < std::min(i_, bSize); j++) {
				attractParticles(i + i_ + regionStart, j + start);
			}
		}

		__m256 ax = _mm256_loadu_ps(x.data + regionStart + i);
		__m256 ay = _mm256_loadu_ps(y.data + regionStart + i);
		__m256 aax = _mm256_loadu_ps(this->ax.data + regionStart + i);
		__m256 aay = _mm256_loadu_ps(this->ay.data + regionStart + i);

		int endIndex = std::max(0, bSize - 8);
		for (int j = 0; j < endIndex; j++) {
			__m256 bx = _mm256_loadu_ps(x.data + start + j);
			__m256 by = _mm256_loadu_ps(y.data + start + j);
			__m256 dx = _mm256_sub_ps(bx, ax);
			__m256 dy = _mm256_sub_ps(by, ay);
			__m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)); // dx*dx + dy*dy

			__m256 invR = _mm256_rsqrt_ps(r2);
			__m256 f = _mm256_mul_ps(ATTRACTION_256, _mm256_mul_ps(invR, _mm256_mul_ps(invR, invR))); // attraction / r^3

			aax = _mm256_fmadd_ps(f, dx, aax);
			aay = _mm256_fmadd_ps(f, dy, aay);
		}

		_mm256_storeu_ps(this->ax.data + regionStart + i, aax);
		_mm256_storeu_ps(this->ay.data + regionStart + i, aay);

		int jMax = bSize - endIndex;
		for (int i_ = 0; i_ < 8; i_++) { // missed by SIMD above
			for (int j_ = i_; j_ < jMax; j_++) {
				attractParticles(i_ + i + regionStart, j_ + endIndex + start);
			}
		}
	}

	for (int i = simdSize; i < regionSize; i++) {
		for (int j = 0; j < bSize; j++) {
			attractParticles(i + regionStart, j + start);
		}
	}
}

void Particles::attractRegion(int region) {
	int start = x.groupStart(region);
	int size = x.groupSize(region);
	int groupedSize = (size / 8) * 8;

	for (int i = 0; i < groupedSize; i += 8) {

		// handle cases where the SIMD regions could overlap, making updating the values weird
		// for example, i = [0, 8), j = [1, 9)
		for (int i_ = 0; i_ < 8; i_++) {
			for (int j_ = i_+1; j_ < std::min(size - i, i_ + 8); j_++) {
				attractParticlesBoth(i_ + i + start, j_ + i + start);
			}
		}

		__m256 ax = _mm256_loadu_ps(x.data + i + start);
		__m256 ay = _mm256_loadu_ps(y.data + i + start);
		__m256 aax = _mm256_loadu_ps(this->ax.data + i + start);
		__m256 aay = _mm256_loadu_ps(this->ay.data + i + start);

		// size-8 to make sure that SIMD doesn't segfault
		for (int j = i + 8; j < size - 8; j++) { // j set to i to avoid duplicate pairs (if we checked (3,5), we don't need to check (5,3))
			__m256 bx = _mm256_loadu_ps(x.data + j + start);
			__m256 by = _mm256_loadu_ps(y.data + j + start);
			__m256 bax = _mm256_loadu_ps(this->ax.data + j + start);
			__m256 bay = _mm256_loadu_ps(this->ay.data + j + start);

			__m256 dx = _mm256_sub_ps(bx, ax);
			__m256 dy = _mm256_sub_ps(by, ay);
			__m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)); // dx*dx + dy*dy

			__m256 invR = _mm256_rsqrt_ps(r2);
			__m256 f = _mm256_mul_ps(ATTRACTION_256, _mm256_mul_ps(invR, _mm256_mul_ps(invR, invR))); // attraction / r^3

			aax = _mm256_fmadd_ps(f, dx, aax);
			aay = _mm256_fmadd_ps(f, dy, aay);
			bax = _mm256_fnmadd_ps(f, dx, bax);
			bay = _mm256_fnmadd_ps(f, dy, bay);

			_mm256_storeu_ps(this->ax.data + j + start, bax);
			_mm256_storeu_ps(this->ay.data + j + start, bay);

		}

		_mm256_storeu_ps(this->ax.data + i + start, aax);
		_mm256_storeu_ps(this->ay.data + i + start, aay);

		int bLast = std::max(i+8, size-8); // where did the previous for loop leave off?
		for (int i_ = 0; i_ < 8; i_++) { // deal with missing items in this i-range because j couldn't get to the end (SIMD needs eight padding)
			for (int j_ = i_ + bLast; j_ < size; j_++) {
				attractParticlesBoth(i_ + i + start, j_ + start);
			}
		}

	}

	// deal with missing items at the end of region, missed because of SIMD grouping
	for (int i = groupedSize; i < size; i++) {
		for (int j = i+1; j < size; j++) {
			attractParticlesBoth(i + start, j + start);
		}
	}
}

void Particles::bounceRegion(int region) {
	int start = x.groupStart(region);
	int size = x.groupSize(region);
	int groupedSize = (size / 8) * 8;

	for (int i = 0; i < groupedSize; i += 8) {

		// handle cases where the SIMD regions could overlap, making updating the values weird
		// for example, i = [0, 8), j = [1, 9)
		for (int i_ = 0; i_ < 8; i_++) {
			for (int j_ = i_+1; j_ < std::min(size - i, i_ + 8); j_++) {
				GLfloat dx = x[j_ + i + start] - x[i_ + i + start];
				GLfloat dy = y[j_ + i + start] - y[i_ + i + start];
				GLfloat r2 = dx*dx + dy*dy;
				if (r2 < PARTICLE_RADIUS2) {
					bounceParticles(i_ + i + start, j_ + i + start, dx, dy, r2);
				}
			}
		}

		__m256 ax = _mm256_loadu_ps(x.data + i + start);
		__m256 ay = _mm256_loadu_ps(y.data + i + start);
		__m256 avx = _mm256_loadu_ps(vx.data + i + start);
		__m256 avy = _mm256_loadu_ps(vy.data + i + start);

		// size-8 to make sure that SIMD doesn't segfault
		for (int j = i + 8; j < size - 8; j++) { // j set to i to avoid duplicate pairs (if we checked (3,5), we don't need to check (5,3))
			__m256 bx = _mm256_loadu_ps(x.data + j + start);
			__m256 by = _mm256_loadu_ps(y.data + j + start);
			__m256 bvx = _mm256_loadu_ps(vx.data + j + start);
			__m256 bvy = _mm256_loadu_ps(vy.data + j + start);

			__m256 dx = _mm256_sub_ps(bx, ax);
			__m256 dy = _mm256_sub_ps(by, ay);
			__m256 dy2 = _mm256_mul_ps(dy, dy);
			__m256 r2 = _mm256_fmadd_ps(dx, dx, dy2);

			__m256 shouldBounce = _mm256_cmp_ps(r2, PARTICLE_RADIUS2_256, _CMP_LT_OQ);
			int anyBounce =  _mm256_movemask_ps(shouldBounce);

			if (anyBounce != 0) {
				__m256 bounceMult = _mm256_and_ps(ENERGY_LOSS_256, shouldBounce);
				__m256 invR = _mm256_rsqrt_ps(r2);

				__m256 ratio = _mm256_mul_ps(invR, PARTICLE_RADIUS_256);
				ratio = _mm256_and_ps(ratio, shouldBounce); // set non-bounced to 0
				ratio = _mm256_or_ps(ratio, _mm256_andnot_ps(shouldBounce, ONE_HALFS_256)); // set non-bounced to 1
				__m256 midX = _mm256_add_ps(ax, bx);
				midX = _mm256_mul_ps(midX, ONE_HALFS_256);
				__m256 midY = _mm256_add_ps(ay, by);
				midY = _mm256_mul_ps(midY, ONE_HALFS_256);

				ax = _mm256_fnmadd_ps(dx, ratio, midX);
				ay = _mm256_fnmadd_ps(dy, ratio, midY);
				bx = _mm256_fmadd_ps(dx, ratio, midX);
				by = _mm256_fmadd_ps(dy, ratio, midY);

				dx = _mm256_mul_ps(dx, invR); // normalize
				dy = _mm256_mul_ps(dy, invR);

				__m256 avAlong = _mm256_fmadd_ps(avx, dx, _mm256_mul_ps(avy, dy)); // avx*dx + avy*dy, A's speed along line bewteen particles
				__m256 bvAlong = _mm256_fmadd_ps(bvx, dx, _mm256_mul_ps(bvy, dy)); // bvx*dx + bvy*dy, B's speed along line bewteen particles
				__m256 avxAlong = _mm256_mul_ps(avAlong, dx); // A's velocity along line bewteen particles
				__m256 avyAlong = _mm256_mul_ps(avAlong, dy);

				__m256 davx = _mm256_fmsub_ps(bvAlong, dx, avxAlong);
				__m256 davy = _mm256_fmsub_ps(bvAlong, dy, avyAlong);
				__m256 dbvx = _mm256_fnmadd_ps(bvAlong, dx, avxAlong);
				__m256 dbvy = _mm256_fnmadd_ps(bvAlong, dy, avyAlong);

				avx = _mm256_fmadd_ps(bounceMult, davx, avx);
				avy = _mm256_fmadd_ps(bounceMult, davy, avy);
				bvx = _mm256_fmadd_ps(bounceMult, dbvx, bvx);
				bvy = _mm256_fmadd_ps(bounceMult, dbvy, bvy);

				_mm256_storeu_ps(x.data + j + start, bx);
				_mm256_storeu_ps(y.data + j + start, by);
				_mm256_storeu_ps(vx.data + j + start, bvx);
				_mm256_storeu_ps(vy.data + j + start, bvy);
			}

		}

		_mm256_storeu_ps(x.data + i + start, ax);
		_mm256_storeu_ps(y.data + i + start, ay);
		_mm256_storeu_ps(vx.data + i + start, avx);
		_mm256_storeu_ps(vy.data + i + start, avy);

		int bLast = std::max(i+8, size-8); // where did the previous for loop leave off?
		for (int i_ = 0; i_ < 8; i_++) { // deal with missing items in this i-range because j couldn't get to the end (SIMD needs eight padding)
			for (int j_ = i_ + bLast; j_ < size; j_++) {
				GLfloat dx = x[j_ + start] - x[i_ + i + start];
				GLfloat dy = y[j_ + start] - y[i_ + i + start];
				GLfloat r2 = dx*dx + dy*dy;
				if (r2 < PARTICLE_RADIUS2) {
					bounceParticles(i_ + i + start, j_ + start, dx, dy, r2);
				}
			}
		}

	}

	// deal with missing items at the end of region, missed because of SIMD grouping
	for (int i = groupedSize; i < size; i++) {
		for (int j = i+1; j < size; j++) {
			GLfloat dx = x[j + start] - x[i + start];
			GLfloat dy = y[j + start] - y[i + start];
			GLfloat r2 = dx*dx + dy*dy;
			if (r2 < PARTICLE_RADIUS2) {
				bounceParticles(i + start, j + start, dx, dy, r2);
			}
		}
	}
}

void Particles::bounceRegions(int regionA, int bStart, int bEnd) { // regionA shouldn't overlap regionB
	int aStart = x.groupStart(regionA);
	int bSize = bEnd - bStart;
	int eightGroupsA = (x.groupSize(regionA) / 8) * 8; // regionA in groups of 8, cut off extras for now

	for (int i = 0; i < eightGroupsA; i += 8) {

		for (int i_ = 0; i_ < 8; i_++) { // missed by SIMD below
			for (int j = 0; j < std::min(i_, bSize); j++) {
				GLfloat dx = x[j + bStart] - x[i_ + i + aStart];
				GLfloat dy = y[j + bStart] - y[i_ + i + aStart];
				GLfloat r2 = dx*dx + dy*dy;
				if (r2 < PARTICLE_RADIUS2) {
					bounceParticles(i_ + i + aStart, j + bStart, dx, dy, r2);
				}
			}
		}

		__m256 ax = _mm256_loadu_ps(x.data + i + aStart);
		__m256 ay = _mm256_loadu_ps(y.data + i + aStart);
		__m256 avx = _mm256_loadu_ps(vx.data + i + aStart);
		__m256 avy = _mm256_loadu_ps(vy.data + i + aStart);

		// make sure that SIMD doesn't segfault
		int endIndex = std::max(0, bSize - 8);
		for (int j = 0; j < endIndex; j++) {
			__m256 bx = _mm256_loadu_ps(x.data + j + bStart);
			__m256 by = _mm256_loadu_ps(y.data + j + bStart);
			__m256 bvx = _mm256_loadu_ps(vx.data + j + bStart);
			__m256 bvy = _mm256_loadu_ps(vy.data + j + bStart);

			__m256 dx = _mm256_sub_ps(bx, ax);
			__m256 dy = _mm256_sub_ps(by, ay);
			__m256 dy2 = _mm256_mul_ps(dy, dy);
			__m256 r2 = _mm256_fmadd_ps(dx, dx, dy2);

			__m256 shouldBounce = _mm256_cmp_ps(r2, PARTICLE_RADIUS2_256, _CMP_LT_OQ);
			int anyBounce =  _mm256_movemask_ps(shouldBounce);

			if (anyBounce != 0) {
				__m256 bounceMult = _mm256_and_ps(ENERGY_LOSS_256, shouldBounce);
				__m256 invR = _mm256_rsqrt_ps(r2);

				__m256 ratio = _mm256_mul_ps(invR, PARTICLE_RADIUS_256);
				ratio = _mm256_and_ps(ratio, shouldBounce); // set non-bounced to 0
				ratio = _mm256_or_ps(ratio, _mm256_andnot_ps(shouldBounce, ONE_HALFS_256)); // set non-bounced to 1
				__m256 midX = _mm256_add_ps(ax, bx);
				midX = _mm256_mul_ps(midX, ONE_HALFS_256);
				__m256 midY = _mm256_add_ps(ay, by);
				midY = _mm256_mul_ps(midY, ONE_HALFS_256);

				ax = _mm256_fnmadd_ps(dx, ratio, midX);
				ay = _mm256_fnmadd_ps(dy, ratio, midY);
				bx = _mm256_fmadd_ps(dx, ratio, midX);
				by = _mm256_fmadd_ps(dy, ratio, midY);

				dx = _mm256_mul_ps(dx, invR); // normalize
				dy = _mm256_mul_ps(dy, invR);

				__m256 avAlong = _mm256_fmadd_ps(avx, dx, _mm256_mul_ps(avy, dy)); // avx*dx + avy*dy, A's speed along line bewteen particles
				__m256 bvAlong = _mm256_fmadd_ps(bvx, dx, _mm256_mul_ps(bvy, dy)); // bvx*dx + bvy*dy, B's speed along line bewteen particles
				__m256 avxAlong = _mm256_mul_ps(avAlong, dx); // A's velocity along line bewteen particles
				__m256 avyAlong = _mm256_mul_ps(avAlong, dy);

				__m256 davx = _mm256_fmsub_ps(bvAlong, dx, avxAlong);
				__m256 davy = _mm256_fmsub_ps(bvAlong, dy, avyAlong);
				__m256 dbvx = _mm256_fnmadd_ps(bvAlong, dx, avxAlong);
				__m256 dbvy = _mm256_fnmadd_ps(bvAlong, dy, avyAlong);

				avx = _mm256_fmadd_ps(bounceMult, davx, avx);
				avy = _mm256_fmadd_ps(bounceMult, davy, avy);
				bvx = _mm256_fmadd_ps(bounceMult, dbvx, bvx);
				bvy = _mm256_fmadd_ps(bounceMult, dbvy, bvy);

				_mm256_storeu_ps(x.data + j + bStart, bx);
				_mm256_storeu_ps(y.data + j + bStart, by);
				_mm256_storeu_ps(vx.data + j + bStart, bvx);
				_mm256_storeu_ps(vy.data + j + bStart, bvy);
			}

		}

		_mm256_storeu_ps(x.data + i + aStart, ax);
		_mm256_storeu_ps(y.data + i + aStart, ay);
		_mm256_storeu_ps(vx.data + i + aStart, avx);
		_mm256_storeu_ps(vy.data + i + aStart, avy);

		int jMax = bSize - endIndex;
		for (int i_ = 0; i_ < 8; i_++) { // missed by SIMD above
			for (int j_ = i_; j_ < jMax; j_++) {
				GLfloat dx = x[j_ + endIndex + bStart] - x[i_ + i + aStart];
				GLfloat dy = y[j_ + endIndex + bStart] - y[i_ + i + aStart];
				GLfloat r2 = dx*dx + dy*dy;
				if (r2 < PARTICLE_RADIUS2) {
					bounceParticles(i_ + i + aStart, j_ + endIndex + bStart, dx, dy, r2);
				}
			}
		}

	}

	// deal with missing items at the end of region A, missed because of SIMD grouping
	for (int i = eightGroupsA; i < x.groupSize(regionA); i++) {
		for (int j = 0; j < bSize; j++) {
			GLfloat dx = x[j + bStart] - x[i + aStart];
			GLfloat dy = y[j + bStart] - y[i + aStart];
			GLfloat r2 = dx*dx + dy*dy;
			if (r2 < PARTICLE_RADIUS2) {
				bounceParticles(i + aStart, j + bStart, dx, dy, r2);
			}
		}
	}
}

void Particles::wallBounce() {
	for(int i = 0; i < PARTICLE_COUNT; i++) {
		if (this->x[i] < PARTICLE_RADIUS) {
			this->x[i] = PARTICLE_RADIUS;
			this->vx[i] = std::abs(this->vx[i]) * ENERGY_LOSS;
		} else if (this->x[i] >= PHYSICS_WIDTH - PARTICLE_RADIUS) {
			this->x[i] = PHYSICS_WIDTH - PARTICLE_RADIUS;
			this->vx[i] = -std::abs(this->vx[i]) * ENERGY_LOSS;
		}
		if (this->y[i] < PARTICLE_RADIUS) {
			this->y[i] = PARTICLE_RADIUS;
			this->vy[i] = std::abs(this->vy[i]) * ENERGY_LOSS;
		} else if (this->y[i] >= PHYSICS_HEIGHT - PARTICLE_RADIUS) {
			this->y[i] = PHYSICS_HEIGHT - PARTICLE_RADIUS;
			this->vy[i] = -std::abs(this->vy[i]) * ENERGY_LOSS;
		}
	}
}

void Particles::draw() const {
	// only ever write to PIXEL_BUFFER_B
	std::fill(PIXEL_BUFFER_B, PIXEL_BUFFER_B + PIXEL_COUNT, 0);

	for (int i = 0; i < PARTICLE_COUNT; i += 8) {

		auto X = _mm256_loadu_ps(this->x.data + i);
		auto Y = _mm256_loadu_ps(this->y.data + i);
		auto xi = _mm256_cvttps_epi32(X); // truncate and convert to int
		auto yi = _mm256_cvttps_epi32(Y); // truncate and convert to int
		xi = _mm256_srli_epi32(xi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE
		yi = _mm256_srli_epi32(yi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE

		int jMax = std::min(i + 8, PARTICLE_COUNT) - i;
		if (DRAW_CIRCLES) {
			for (int j = 0; j < jMax; j++) {
				int x = ((int32_t*)&xi)[j];
				int y = ((int32_t*)&yi)[j];
				for (int y_ = std::max(0, y - PARTICLE_RADIUS/PHYSICS_SCALE); y_ < std::min(WINDOW_HEIGHT, y + PARTICLE_RADIUS/PHYSICS_SCALE); y_++) {
					for (int x_ = std::max(0, x - PARTICLE_RADIUS/PHYSICS_SCALE); x_ < std::min(WINDOW_WIDTH, x + PARTICLE_RADIUS/PHYSICS_SCALE); x_++) {
						if ((y-y_)*(y-y_) + (x-x_)*(x-x_) < PARTICLE_RADIUS*PARTICLE_RADIUS/PHYSICS_SCALE/PHYSICS_SCALE) {
							PIXEL_BUFFER_B[y_*WINDOW_WIDTH + x_] = 255;
						}
					}
				}
			}
		} else {
			yi = _mm256_mullo_epi32(yi, WINDOW_WIDTH_256); // y*WINDOW_WIDTH
			yi = _mm256_add_epi32(yi, xi); // y*WINDOW_WIDTH + x
			int32_t* Is = (int32_t*) &yi;
			for (int j = 0; j < jMax; j++) {
				PIXEL_BUFFER_B[Is[j]] = 255;
			}
		}
		
	}

	swapMutex.lock();
	std::swap(PIXEL_BUFFER_A, PIXEL_BUFFER_B);
	swapMutex.unlock();
}

int Particles::particleRegion(GLfloat x, GLfloat y) const {
	return static_cast<int>(y / REGION_HEIGHT)*REGIONS_ACROSS + static_cast<int>(x / REGION_WIDTH);
}

int Particles::regionIndex(int i, int j) const {
	return j*REGIONS_ACROSS + i;
}

float Particles::rsqrt_fast(float x) const {
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
}