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

	struct IndexPair {
		uint16_t i;
		uint16_t j;
	};

	std::chrono::steady_clock timer;
	std::chrono::steady_clock::time_point lastTime;
	bool hasTime;

	long tickCount;
	double dtTotal;

	void tickBounceAndAttractSIMD();
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

Particles::Particles(int count) {
	this->x = (GLfloat*)_aligned_malloc(count*4, 256);
	this->y = (GLfloat*)_aligned_malloc(count*4, 256);
	this->vx = (GLfloat*)_aligned_malloc(count*4, 256);
	this->vy = (GLfloat*)_aligned_malloc(count*4, 256);
	this->ax = (GLfloat*)_aligned_malloc(count*4, 256);
	this->ay = (GLfloat*)_aligned_malloc(count*4, 256);

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
	__m256i tmp = _mm256_set1_epi32(0x7FFFFFFF);
	FLOAT_ABS_256 = *(__m256*)&tmp;
	ENERGY_LOSS_256 = _mm256_set1_ps(ENERGY_LOSS);
	
	WINDOW_WIDTH_256 = _mm256_set1_epi32(WINDOW_WIDTH);
	ZEROS_256 = _mm256_set1_epi32(0);

	int32_t a[8] = {7, 0, 1, 2, 3, 4, 5, 6};
	ROTATE_RIGHT_256 = _mm256_loadu_si256((__m256i*) &a);
}

Particles::~Particles() {
	delete[] this->x;
	delete[] this->y;
	delete[] this->vx;
	delete[] this->vy;
	delete[] this->ax;
	delete[] this->ay;
}


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

	float dts[8] = {dt, dt, dt, dt, dt, dt, dt, dt};
	__m256 DT_256 = _mm256_load_ps(dts);

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
		GLfloat x0 = this->x[i+7];
		GLfloat y0 = this->y[i+7];
		int x0i = static_cast<int>(std::floor(x0)) >> PHYSICS_SCALE_POWER;
		int y0i = static_cast<int>(std::floor(y0)) >> PHYSICS_SCALE_POWER;
		int bufferI = y0i*WINDOW_WIDTH + x0i;

		auto X = _mm256_load_ps(this->x + i);
		auto Y = _mm256_load_ps(this->y + i);
		auto xi = _mm256_cvttps_epi32(X);
		auto yi = _mm256_cvttps_epi32(Y);
		xi = _mm256_srli_epi32(xi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE
		yi = _mm256_srli_epi32(yi, PHYSICS_SCALE_POWER); // divide by PHYSICS_SCALE
		// yi = _mm256_mullo_epi32(yi, WINDOW_WIDTH_256); // y*WINDOW_WIDTH
		// yi = _mm256_add_epi32(yi, xi); // y*WINDOW_WIDTH + x

		// int32_t* Is = (int32_t*) &yi;
		for (int j = 0; j < 8; j++) {
			int x = ((int32_t*)&xi)[j];
			int y = ((int32_t*)&yi)[j];
			for (int y_ = std::max(0, y - PARTICLE_RADIUS/PHYSICS_SCALE); y_ < std::min(WINDOW_HEIGHT, y + PARTICLE_RADIUS/PHYSICS_SCALE); y_++) {
				for (int x_ = std::max(0, x - PARTICLE_RADIUS/PHYSICS_SCALE); x_ < std::min(WINDOW_WIDTH, x + PARTICLE_RADIUS/PHYSICS_SCALE); x_++) {
					if ((y-y_)*(y-y_) + (x-x_)*(x-x_) < PARTICLE_RADIUS*PARTICLE_RADIUS/PHYSICS_SCALE/PHYSICS_SCALE) {
						PIXEL_BUFFER_B[y_*WINDOW_WIDTH + x_] = 255;
					}
				}
			}
			// PIXEL_BUFFER_B[Is[j]] = 255;
		}
	}

	std::swap(PIXEL_BUFFER_A, PIXEL_BUFFER_B);
}

void Particles::tickBounceAndAttractSIMD() {

	GLfloat indexArray[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
	for (int i = 0; i < PARTICLE_COUNT; i += 8) {
		auto ax = _mm256_load_ps(this->x + i);
		auto ay = _mm256_load_ps(this->y + i);
		auto avx = _mm256_load_ps(this->vx + i);
		auto avy = _mm256_load_ps(this->vy + i);
		auto aax = _mm256_load_ps(this->ax + i);
		auto aay = _mm256_load_ps(this->ay + i);

		auto bx = ax;
		auto by = ay;
		auto bvx = avx;
		auto bvy = avy;
		auto bax = aax;
		auto bay = aay;

		auto bIndex = _mm256_loadu_ps(indexArray);
		auto indexOrig = _mm256_loadu_ps(indexArray);
		for (int k = 0; k < 7; k++) {
			bx = _mm256_permutevar8x32_ps(bx, ROTATE_RIGHT_256);
			by = _mm256_permutevar8x32_ps(by, ROTATE_RIGHT_256);
			bvx = _mm256_permutevar8x32_ps(bvx, ROTATE_RIGHT_256);
			bvy = _mm256_permutevar8x32_ps(bvy, ROTATE_RIGHT_256);
			bax = _mm256_permutevar8x32_ps(bax, ROTATE_RIGHT_256);
			bay = _mm256_permutevar8x32_ps(bay, ROTATE_RIGHT_256);
			bIndex = _mm256_permutevar8x32_ps(bIndex, ROTATE_RIGHT_256);

			auto dx = _mm256_sub_ps(bx, ax);
			auto dy = _mm256_sub_ps(by, ay);
			auto dx2 = _mm256_mul_ps(dx, dx); // dx^2

			auto r2 = _mm256_fmadd_ps(dy, dy, dx2); // dx^2 + dy^2
			auto r2Greater = _mm256_cmp_ps(r2, PARTICLE_RADIUS2_256, _CMP_GE_OQ); // 0xFFFFFFFF if outside, 0 if inside

			auto invR = _mm256_rsqrt_ps(r2); // 1 / sqrt(r^2) => 1/r
			auto invR2 = _mm256_rcp_ps(r2); // 1 / r^2
			invR2 = _mm256_mul_ps(invR, invR2); // 1/r * 1/r^2 = 1/r^3
			invR2 = _mm256_and_ps(invR2, r2Greater); // mask out those within particle
			auto invR_ATT = _mm256_mul_ps(ATTRACTION_256, invR2);
			
			aax = _mm256_fmadd_ps(dx, invR_ATT, aax);
			aay = _mm256_fmadd_ps(dy, invR_ATT, aay);
			bax = _mm256_fnmadd_ps(dx, invR_ATT, bax);
			bay = _mm256_fnmadd_ps(dy, invR_ATT, bay);


			// begin bounce code

			auto mask = _mm256_cmp_ps(bIndex, indexOrig, _CMP_GT_OQ); // only do for index of b > index of a
			mask = _mm256_andnot_ps(r2Greater, mask);
			for (int i = 0; i < 8; i++) {
				if (((uint32_t*)&mask)[i] != 0) {
					printf("cum\n");
				}
			}

			auto centerX = _mm256_add_ps(ax, bx);
			auto centerY = _mm256_add_ps(ay, by);
			centerX = _mm256_mul_ps(centerX, ONE_HALFS_256);
			centerY = _mm256_mul_ps(centerY, ONE_HALFS_256);

			auto dAx = _mm256_sub_ps(ax, centerX);
			auto dAy = _mm256_sub_ps(ay, centerY);
			auto dAlength = _mm256_fmadd_ps(dAx, dAx, _mm256_mul_ps(dAy, dAy));
			dAlength = _mm256_rsqrt_ps(dAlength);

			auto scale = _mm256_andnot_ps(mask, ONES_256); // ones where a doesn't overlap b
			auto ratio = _mm256_and_ps(mask, _mm256_mul_ps(PARTICLE_RADIUS_256, dAlength)); // ratio where a does overlap b
			scale = _mm256_or_ps(scale, ratio); // combine

			ax = _mm256_fmadd_ps(scale, dAx, centerX);
			ay = _mm256_fmadd_ps(scale, dAy, centerY);
			bx = _mm256_fnmadd_ps(scale, dAx, centerX);
			by = _mm256_fnmadd_ps(scale, dAy, centerY);


			dx = _mm256_mul_ps(dx, invR); // normalize
			dy = _mm256_mul_ps(dy, invR); // normalize

			auto avAlongPerp = _mm256_fmadd_ps(avx, dx, _mm256_mul_ps(avy, dy)); // avx*dx + avy*dy
			auto bvAlongPerp = _mm256_fmadd_ps(bvx, dx, _mm256_mul_ps(bvy, dy)); // bvx*dx + bvy*dy
			auto avAlongPerpX = _mm256_mul_ps(avAlongPerp, dx);
			auto avAlongPerpY = _mm256_mul_ps(avAlongPerp, dy);

			auto dvx = _mm256_fmsub_ps(dx, bvAlongPerp, avAlongPerpX);
			auto dvy = _mm256_fmsub_ps(dy, bvAlongPerp, avAlongPerpY);

			mask = _mm256_and_ps(mask, ENERGY_LOSS_256);
			avx = _mm256_fmadd_ps(dvx, mask, avx);
			avy = _mm256_fmadd_ps(dvy, mask, avy);
			bvx = _mm256_fnmadd_ps(dvx, mask, bvx);
			bvy = _mm256_fnmadd_ps(dvy, mask, bvy);

		}

		// extra rotate to align b normally
		bvx = _mm256_permutevar8x32_ps(bvx, ROTATE_RIGHT_256);
		bvy = _mm256_permutevar8x32_ps(bvy, ROTATE_RIGHT_256);
		bax = _mm256_permutevar8x32_ps(bax, ROTATE_RIGHT_256);
		bay = _mm256_permutevar8x32_ps(bay, ROTATE_RIGHT_256);

		_mm256_store_ps(this->x + i, bx);
		_mm256_store_ps(this->y + i, by);
		_mm256_store_ps(this->vx + i, bvx);
		_mm256_store_ps(this->vy + i, bvy);
		_mm256_store_ps(this->ax + i, bax);
		_mm256_store_ps(this->ay + i, bay);

		ax = bx;
		ay = by;
		avx = bvx;
		avy = bvy;
		aax = bax;
		aay = bay;

		for (int j = i+8; j < PARTICLE_COUNT; j += 8) {
			auto bx = _mm256_load_ps(this->x + j);
			auto by = _mm256_load_ps(this->y + j);
			auto bvx = _mm256_load_ps(this->vx + j);
			auto bvy = _mm256_load_ps(this->vy + j);
			auto bax = _mm256_load_ps(this->ax + j);
			auto bay = _mm256_load_ps(this->ay + j);

			for (int k = 0; k < 8; k++) {
				bx = _mm256_permutevar8x32_ps(bx, ROTATE_RIGHT_256);
				by = _mm256_permutevar8x32_ps(by, ROTATE_RIGHT_256);
				bvx = _mm256_permutevar8x32_ps(bvx, ROTATE_RIGHT_256);
				bvy = _mm256_permutevar8x32_ps(bvy, ROTATE_RIGHT_256);
				bax = _mm256_permutevar8x32_ps(bax, ROTATE_RIGHT_256);
				bay = _mm256_permutevar8x32_ps(bay, ROTATE_RIGHT_256);

				auto dx = _mm256_sub_ps(bx, ax);
				auto dy = _mm256_sub_ps(by, ay);
				auto dx2 = _mm256_mul_ps(dx, dx); // dx^2

				auto r2 = _mm256_fmadd_ps(dy, dy, dx2); // dx^2 + dy^2
				auto r2Greater = _mm256_cmp_ps(r2, PARTICLE_RADIUS2_256, _CMP_GE_OQ); // 0xFFFFFFFF if outside, 0 if inside

				auto invR = _mm256_rsqrt_ps(r2); // 1 / sqrt(r^2) => 1/r
				auto invR2 = _mm256_rcp_ps(r2); // 1 / r^2
				invR2 = _mm256_mul_ps(invR, invR2); // 1/r * 1/r^2 = 1/r^3
				invR2 = _mm256_and_ps(invR2, r2Greater); // mask out those within particle
				auto invR_ATT = _mm256_mul_ps(ATTRACTION_256, invR2);
				
				aax = _mm256_fmadd_ps(dx, invR_ATT, aax);
				aay = _mm256_fmadd_ps(dy, invR_ATT, aay);
				bax = _mm256_fnmadd_ps(dx, invR_ATT, bax);
				bay = _mm256_fnmadd_ps(dy, invR_ATT, bay);


				// begin bounce code

				auto centerX = _mm256_add_ps(ax, bx);
				auto centerY = _mm256_add_ps(ay, by);
				centerX = _mm256_mul_ps(centerX, ONE_HALFS_256);
				centerY = _mm256_mul_ps(centerY, ONE_HALFS_256);

				auto dAx = _mm256_sub_ps(ax, centerX);
				auto dAy = _mm256_sub_ps(ay, centerY);
				auto dAlength = _mm256_fmadd_ps(dAx, dAx, _mm256_mul_ps(dAy, dAy));
				dAlength = _mm256_rsqrt_ps(dAlength);

				auto scale = _mm256_and_ps(r2Greater, ONES_256); // ones where a doesn't overlap b
				auto ratio = _mm256_andnot_ps(r2Greater, _mm256_mul_ps(PARTICLE_RADIUS_256, dAlength)); // ratio where a does overlap b
				scale = _mm256_or_ps(scale, ratio); // combine

				ax = _mm256_fmadd_ps(scale, dAx, centerX);
				ay = _mm256_fmadd_ps(scale, dAy, centerY);
				bx = _mm256_fnmadd_ps(scale, dAx, centerX);
				by = _mm256_fnmadd_ps(scale, dAy, centerY);


				dx = _mm256_mul_ps(dx, invR); // normalize
				dy = _mm256_mul_ps(dy, invR); // normalize

				auto avAlongPerp = _mm256_fmadd_ps(avx, dx, _mm256_mul_ps(avy, dy)); // avx*dx + avy*dy
				auto bvAlongPerp = _mm256_fmadd_ps(bvx, dx, _mm256_mul_ps(bvy, dy)); // bvx*dx + bvy*dy
				auto avAlongPerpX = _mm256_mul_ps(avAlongPerp, dx);
				auto avAlongPerpY = _mm256_mul_ps(avAlongPerp, dy);

				auto dvx = _mm256_fmsub_ps(dx, bvAlongPerp, avAlongPerpX);
				auto dvy = _mm256_fmsub_ps(dy, bvAlongPerp, avAlongPerpY);

				auto energy = _mm256_andnot_ps(r2Greater, ENERGY_LOSS_256);
				avx = _mm256_fmadd_ps(dvx, energy, avx);
				avy = _mm256_fmadd_ps(dvy, energy, avy);
				bvx = _mm256_fnmadd_ps(dvx, energy, bvx);
				bvy = _mm256_fnmadd_ps(dvy, energy, bvy);

			}

			_mm256_store_ps(this->x + j, bx);
			_mm256_store_ps(this->y + j, by);
			_mm256_store_ps(this->vx + j, bvx);
			_mm256_store_ps(this->vy + j, bvy);
			_mm256_store_ps(this->ax + j, bax);
			_mm256_store_ps(this->ay + j, bay);
		}

		_mm256_store_ps(this->x + i, ax);
		_mm256_store_ps(this->y + i, ay);
		_mm256_store_ps(this->vx + i, avx);
		_mm256_store_ps(this->vy + i, avy);
		_mm256_store_ps(this->ax + i, aax);
		_mm256_store_ps(this->ay + i, aay);

	}

}