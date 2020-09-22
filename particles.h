#include <glad/glad.h>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <immintrin.h>

#include "constants.h"

class Particles {
public:
	GLfloat* x;
	GLfloat* y;
	GLfloat* vx;
	GLfloat* vy;
	GLfloat* ax;
	GLfloat* ay;

	Particles(int count);
	~Particles();
	void tick();

private:
	std::chrono::steady_clock timer;
	std::chrono::steady_clock::time_point lastTime;
	bool hasTime;

	long tickCount;
	double dtTotal;

	float rsqrt_fast(float x) const;

	void tickBounceAndAttractSIMD();
	void tickBounceAndAttract();

	__m256 PARTICLE_RADIUS2_256;
	__m256 ATTRACTION_256;
	__m256i ROTATE_RIGHT_256;
	__m256i WINDOW_WIDTH_256;

};

Particles::Particles(int count) {
	this->x = new GLfloat[count];
	this->y = new GLfloat[count];
	this->vx = new GLfloat[count];
	this->vy = new GLfloat[count];
	this->ax = new GLfloat[count];
	this->ay = new GLfloat[count];

	hasTime = false;
	tickCount = 0;
	dtTotal = 0;

	float a[8];
	for (int i = 0; i < 8; i++) {
		a[i] = PARTICLE_RADIUS2;
	}
	PARTICLE_RADIUS2_256 = _mm256_loadu_ps(a);

	for (int i = 0; i < 8; i++) {
		a[i] = ATTRACTION;
	}
	ATTRACTION_256 = _mm256_loadu_ps(a);

	int32_t b[8] = {7, 0, 1, 2, 3, 4, 5, 6};
	ROTATE_RIGHT_256 = _mm256_loadu_si256((__m256i*) &b);

	for (int i = 0; i < 8; i++) {
		b[i] = WINDOW_WIDTH;
	}
	WINDOW_WIDTH_256 = _mm256_loadu_si256((__m256i*) &b);
}

Particles::~Particles() {
	delete[] this->x;
	delete[] this->y;
	delete[] this->vx;
	delete[] this->vy;
	delete[] this->ax;
	delete[] this->ay;
};

__forceinline float Particles::rsqrt_fast(float x) const {
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
}

void Particles::tick() { // only ever write to PIXEL_BUFFER_B
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

	// float dts[8] = {dt, dt, dt, dt, dt, dt, dt, dt};
	// __m256 DT_256 = _mm256_loadu_ps(dts);

	for (int i = 0; i < PARTICLE_COUNT; i++) {
		this->vx[i] += this->ax[i] * dt;
		this->vy[i] += this->ay[i] * dt;
		this->x[i] += this->vx[i] * dt;
		this->y[i] += this->vy[i] * dt;
		this->ax[i] = 0;
		this->ay[i] = -GRAVITY;
	}

	this->tickBounceAndAttractSIMD();

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

	std::fill(PIXEL_BUFFER_B, PIXEL_BUFFER_B + PIXEL_COUNT, 0);
	for (int i = 0; i < PARTICLE_COUNT; i += 8) {
		auto x = _mm256_loadu_ps(this->x + i);
		auto y = _mm256_loadu_ps(this->y + i);
		auto xi = _mm256_cvttps_epi32(x);
		auto yi = _mm256_cvttps_epi32(y);
		xi = _mm256_srli_epi32(xi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE
		yi = _mm256_srli_epi32(yi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE
		yi = _mm256_mul_epi32(yi, WINDOW_WIDTH_256); // y*WINDOW_WIDTH
		yi = _mm256_add_epi32(yi, xi); // y*WINDOW_WIDTH + x

		int32_t* Is = (int32_t*) &yi;
		for (int j = 0; j < 8; j++) {
			PIXEL_BUFFER_B[Is[j]] = 255;
		}
	}

	std::swap(PIXEL_BUFFER_A, PIXEL_BUFFER_B);
}

void Particles::tickBounceAndAttractSIMD() {

	for (int i = 0; i < PARTICLE_COUNT; i += 8) {
		__m256 ax = _mm256_loadu_ps(this->x + i);
		__m256 ay = _mm256_loadu_ps(this->y + i);
		__m256 aax = _mm256_loadu_ps(this->ax + i);
		__m256 aay = _mm256_loadu_ps(this->ay + i);

		for (int j = 0; j < PARTICLE_COUNT; j += 8) {
			__m256 bx = _mm256_loadu_ps(this->x + j);
			__m256 by = _mm256_loadu_ps(this->y + j);

			int count = (i == j) ? 7 : 8;
			for (int k = 0; k < count; k++) {
				bx = _mm256_permutevar8x32_ps(bx, ROTATE_RIGHT_256);
				by = _mm256_permutevar8x32_ps(by, ROTATE_RIGHT_256);

				auto dx = _mm256_sub_ps(bx, ax);
				auto dy = _mm256_sub_ps(by, ay);
				auto dx2 = _mm256_mul_ps(dx, dx); // dx^2

				auto r2 = _mm256_fmadd_ps(dy, dy, dx2); // dx^2 + dy^2
				auto r2Greater = _mm256_cmp_ps(r2, PARTICLE_RADIUS2_256, _CMP_GE_OQ); // check which are outside particle

				auto invR = _mm256_rsqrt_ps(r2); // 1 / sqrt(r^2) => 1/r
				auto invR2 = _mm256_rcp_ps(r2); // 1 / r^2
				invR = _mm256_mul_ps(invR, invR2); // 1/r * 1/r^2 = 1/r^3
				invR = _mm256_and_ps(invR, r2Greater); // mask out those within particle
				auto invR_ATT = _mm256_mul_ps(ATTRACTION_256, invR);
				
				aax = _mm256_fmadd_ps(dx, invR_ATT, aax);
				aay = _mm256_fmadd_ps(dy, invR_ATT, aay);

			}
		}

		_mm256_storeu_ps(this->ax + i, aax);
		_mm256_storeu_ps(this->ay + i, aay);

	}
}

void Particles::tickBounceAndAttract() {
	for (int i = 0; i < PARTICLE_COUNT; i++) {

		GLfloat ax = this->x[i];
		GLfloat ay = this->y[i];
		GLfloat aax = this->ax[i];
		GLfloat aay = this->ay[i];
		// bool collided = false;

		for (int j = 0; j < PARTICLE_COUNT; j++) {
			GLfloat dx = ax - this->x[j];
			GLfloat dy = ay - this->y[j];
			GLfloat r2 = dx*dx + dy*dy;

			if (r2 >= PARTICLE_RADIUS2) {
				r2 = rsqrt_fast(r2);
				r2 = r2*r2*r2;
				aax += -ATTRACTION * dx*r2;
				aay += -ATTRACTION * dy*r2;
			} // else if (!collided) {
				// GLfloat avx = this->vx[i];
				// GLfloat avy = this->vy[i];

				// GLfloat adx = std::abs(dx);
				// GLfloat ady = std::abs(dy);
				// GLfloat dx_, dy_;
				// if (ady != 0) {
				// 	dy_ = std::sqrt(PARTICLE_RADIUS2 / (1 + (dx*dx)/(dy*dy)));
				// 	dx_ = adx / ady * dy_;
				// } else if (adx != 0) {
				// 	dx_ = std::sqrt(PARTICLE_RADIUS2 / (1 + (dy*dy)/(dx*dx)));
				// 	dy_ = ady / adx * dx_;
				// } else {
				// 	continue;
				// }
				// collided = true;

				// GLfloat ddx = dx_ - adx;
				// GLfloat ddy = dy_ - ady;
				// if (ax < this->x[j]) {
				// 	ax -= ddx/2;
				// 	this->x[j] += ddx/2;
				// } else {
				// 	ax += ddx/2;
				// 	this->x[j] -= ddx/2;
				// }
				// if (ay < this->y[j]) {
				// 	ay -= ddy/2;
				// 	this->y[j] += ddy/2;
				// } else {
				// 	ay += ddy/2;
				// 	this->y[j] -= ddy/2;
				// }

				// GLfloat tanX = ay - this->y[j];
				// GLfloat tanY = this->x[j] - ax;
				// GLfloat tanLength = std::sqrt(tanX*tanX + tanY*tanY);
				// tanX /= tanLength;
				// tanY /= tanLength;

				// GLfloat avAlongTan = avx * tanX + avy * tanY;
				// GLfloat avPerpTanX = avx - avAlongTan * tanX;
				// GLfloat avPerpTanY = avy - avAlongTan * tanY;
				// GLfloat bvAlongTan = this->vx[j] * tanX + this->vy[j] * tanY;
				// GLfloat bvPerpTanX = this->vx[j] - bvAlongTan * tanX;
				// GLfloat bvPerpTanY = this->vy[j] - bvAlongTan * tanY;

				// this->vx[i] = (avAlongTan * tanX + bvPerpTanX) * ENERGY_LOSS;
				// this->vy[i] = (avAlongTan * tanY + bvPerpTanY) * ENERGY_LOSS;
				// this->vx[j] = (bvAlongTan * tanX + avPerpTanX) * ENERGY_LOSS;
				// this->vy[j] = (bvAlongTan * tanY + avPerpTanY) * ENERGY_LOSS;
			// }
		}

		this->x[i] = ax;
		this->y[i] = ay;
		this->ax[i] = aax;
		this->ay[i] = aay;

	}
}

/*
for (int y_ = std::max(0, y - PARTICLE_RADIUS/PHYSICS_SCALE); y_ < std::min(WINDOW_HEIGHT, y + PARTICLE_RADIUS/PHYSICS_SCALE); y_++) {
	for (int x_ = std::max(0, x - PARTICLE_RADIUS/PHYSICS_SCALE); x_ < std::min(WINDOW_WIDTH, x + PARTICLE_RADIUS/PHYSICS_SCALE); x_++) {
		
	}
}
*/