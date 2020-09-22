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
	__m256 PARTICLE_DIAMETER_256;
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

	int* bounceIndecies;
	IndexPair* pairList;
	int** indexTable;

};

Particles::Particles(int count) {
	this->x = (GLfloat*)_aligned_malloc(count*4, 256);
	this->y = (GLfloat*)_aligned_malloc(count*4, 256);
	this->vx = (GLfloat*)_aligned_malloc(count*4, 256);
	this->vy = (GLfloat*)_aligned_malloc(count*4, 256);
	this->ax = (GLfloat*)_aligned_malloc(count*4, 256);
	this->ay = (GLfloat*)_aligned_malloc(count*4, 256);
	this->bounceIndecies = (int*)_aligned_malloc(count*4, 256);
	this->pairList = new IndexPair[count];
	this->indexTable = new int*[8];
	for (int i = 0; i < 8; i++) {
		this->indexTable[i] = (int*)_aligned_malloc(8*4, 256);
	}
	

	hasTime = false;
	tickCount = 0;
	dtTotal = 0;

	PARTICLE_RADIUS2_256 = _mm256_set1_ps(PARTICLE_RADIUS2);
	PARTICLE_DIAMETER_256 = _mm256_set1_ps(PARTICLE_RADIUS*2);
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

	for (int k = 0; k < 8; k++) {
		for (int n = 0; n < 8; n++) {
			indexTable[k][n] = ((n + 15 - k) % 8) + 1;
		}
	}
}

Particles::~Particles() {
	delete[] this->x;
	delete[] this->y;
	delete[] this->vx;
	delete[] this->vy;
	delete[] this->ax;
	delete[] this->ay;
	delete[] this->bounceIndecies;
	delete[] this->pairList;

	for (int i = 0; i < 8; i++) {
		delete[] this->indexTable[i];
	}
	delete[] this->indexTable;
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

	for (int i = 0; i < PARTICLE_COUNT; i += 8) {
		auto ax = _mm256_load_ps(this->x + i);
		auto ay = _mm256_load_ps(this->y + i);
		auto avx = _mm256_load_ps(this->vx + i);
		auto avy = _mm256_load_ps(this->vy + i);
		auto aax = _mm256_load_ps(this->ax + i);
		auto aay = _mm256_load_ps(this->ay + i);
		auto indexSection = _mm256_srai_epi32(ZEROS_256, 0);

		for (int j = i; j < PARTICLE_COUNT; j += 8) {
			auto bx = _mm256_load_ps(this->x + j);
			auto by = _mm256_load_ps(this->y + j);
			auto bvx = _mm256_load_ps(this->vx + j);
			auto bvy = _mm256_load_ps(this->vy + j);

			int count = 8 - (i == j);
			for (int k = 0; k < count; k++) {
				auto indexTableSection = _mm256_load_si256((__m256i*)(indexTable[k]));
				indexTableSection = _mm256_add_epi32(indexTableSection, _mm256_set1_epi32(j));
				bx = _mm256_permutevar8x32_ps(bx, ROTATE_RIGHT_256);
				by = _mm256_permutevar8x32_ps(by, ROTATE_RIGHT_256);
				bvx = _mm256_permutevar8x32_ps(bvx, ROTATE_RIGHT_256);
				bvy = _mm256_permutevar8x32_ps(bvy, ROTATE_RIGHT_256);

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

				auto mask = _mm256_or_si256(*(__m256i*)&r2Greater, indexSection);
				mask = _mm256_cmpeq_epi32(mask, ZEROS_256);

				auto nonZeroR2 = _mm256_cmp_ps(r2, ZEROSF_256, _CMP_NEQ_OQ);
				mask = _mm256_and_si256(mask, *(__m256i*)&nonZeroR2); // don't try to bounce off distance == 0 particles

				auto newIndecies = _mm256_and_si256(indexTableSection, mask);
				indexSection = _mm256_or_si256(indexSection, newIndecies);

				dx = _mm256_mul_ps(dx, invR); // normalize
				dy = _mm256_mul_ps(dy, invR); // normalize

				auto avAlongTan = _mm256_fnmadd_ps(avy, dx, _mm256_mul_ps(avx, dy)); // avy*-dx + avx*dy
				auto avAlongTanX = _mm256_mul_ps(avAlongTan, dy);
				auto avAlongPerpX = _mm256_sub_ps(avx, avAlongTanX);
				auto negAvAlongTanY = _mm256_mul_ps(avAlongTan, dx);
				auto avAlongPerpY = _mm256_add_ps(avy, negAvAlongTanY);

				auto bvAlongTan = _mm256_fnmadd_ps(bvy, dx, _mm256_mul_ps(bvx, dy)); // bvy*-dx + bvx*dy
				auto bvAlongTanX = _mm256_mul_ps(bvAlongTan, dy);
				auto bvAlongPerpX = _mm256_sub_ps(bvx, bvAlongTanX);
				auto negBvAlongTanY = _mm256_mul_ps(bvAlongTan, dx);
				auto bvAlongPerpY = _mm256_add_ps(bvy, negBvAlongTanY);

				auto davx = _mm256_add_ps(avAlongTanX, bvAlongPerpX);
				auto davy = _mm256_sub_ps(bvAlongPerpY, negAvAlongTanY);
				auto dbvx = _mm256_add_ps(bvAlongTanX, avAlongPerpX);
				auto dbvy = _mm256_sub_ps(avAlongPerpY, negBvAlongTanY);

				avx = _mm256_fmadd_ps(davx, ENERGY_LOSS_256, avx);
				avy = _mm256_fmadd_ps(davy, ENERGY_LOSS_256, avx);
				bvx = _mm256_fmadd_ps(dbvx, ENERGY_LOSS_256, bvx);
				bvy = _mm256_fmadd_ps(dbvy, ENERGY_LOSS_256, bvx);

			}

		}

		_mm256_store_si256((__m256i*)(this->bounceIndecies + i), indexSection);
		_mm256_store_ps(this->ax + i, aax);
		_mm256_store_ps(this->ay + i, aay);

	}

	int pairCount = 0;
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		int j = this->bounceIndecies[i];
		if (j > i) {
			pairList[pairCount].i = i;
			pairList[pairCount].j = j - 1;
			pairCount++;
		}
	}

	GLfloat data[8][8];
	for (int n = 0; n < pairCount; n += 8) {

		int remaining = std::min(pairCount, n+8) - n;
		for (int k = 0; k < remaining; k++) {
			data[0][k] = this->x[pairList[n+k].i];
			data[1][k] = this->y[pairList[n+k].i];
			data[2][k] = this->vx[pairList[n+k].i];
			data[3][k] = this->vy[pairList[n+k].i];
			data[4][k] = this->x[pairList[n+k].j];
			data[5][k] = this->y[pairList[n+k].j];
			data[6][k] = this->vx[pairList[n+k].j];
			data[7][k] = this->vy[pairList[n+k].j];
		}

		auto ax = _mm256_loadu_ps(data[0]);
		auto ay = _mm256_loadu_ps(data[1]);
		auto avx = _mm256_loadu_ps(data[2]);
		auto avy = _mm256_loadu_ps(data[3]);

		auto bx = _mm256_loadu_ps(data[4]);
		auto by = _mm256_loadu_ps(data[5]);
		auto bvx = _mm256_loadu_ps(data[6]);
		auto bvy = _mm256_loadu_ps(data[7]);


		auto dxOrig = _mm256_sub_ps(ax, bx);
		auto dx = _mm256_and_ps(dxOrig, FLOAT_ABS_256); // abs( dxOrig ) by setting MSB of float to 0
		auto dyOrig = _mm256_sub_ps(ay, by);
		auto dy = _mm256_and_ps(dyOrig, FLOAT_ABS_256); // abs( dyOrig ) by setting MSB of float to 0

		auto dx2 = _mm256_mul_ps(dx, dx);
		auto dy2 = _mm256_mul_ps(dy, dy);

		auto dyNotZero = _mm256_cmp_ps(dy, ZEROSF_256, _CMP_NEQ_OQ);
		auto xOverY = _mm256_div_ps(dx, dy);
		auto yOverX = _mm256_div_ps(dy, dx);

		xOverY = _mm256_and_ps(dyNotZero, xOverY); // only have ratio != 0 where it is non-inf
		yOverX = _mm256_andnot_ps(dyNotZero, yOverX); // anywhere that dy == 0, dx != 0 because of `don't try to bounce off distance == 0 particles`

		auto xyRatio = _mm256_or_ps(xOverY, yOverX); // ratio should be defined everywhere (unless dx == 0 && dy == 0)
		auto xyRatio2 = _mm256_mul_ps(xyRatio, xyRatio);


		auto d1 = _mm256_add_ps(xyRatio2, ONES_256); 
		d1 = _mm256_rsqrt_ps(d1);
		d1 = _mm256_mul_ps(d1, PARTICLE_DIAMETER_256); // 1/sqrt(1 + ratio^2) * diameter
		auto d2 = _mm256_mul_ps(d1, xyRatio); // d1 * ratio

		auto dy_ = _mm256_and_ps(d1, dyNotZero); // where dy != 0, this is the correct formula
		auto dx_ = _mm256_and_ps(d2, dyNotZero);

		dy_ = _mm256_or_ps(dy_, _mm256_andnot_ps(dyNotZero, d2)); // where dy == 0, d1 and d2 should be swapped
		dx_ = _mm256_or_ps(dx_, _mm256_andnot_ps(dyNotZero, d1));

		auto ddx = _mm256_sub_ps(dx_, dx);
		auto ddy = _mm256_sub_ps(dy_, dy);

		auto aLessX = _mm256_cmp_ps(ax, bx, _CMP_LT_OQ);
		auto aLessY = _mm256_cmp_ps(ay, by, _CMP_LT_OQ);

		auto invertMultX = _mm256_and_ps(aLessX, NEGATIVE_ONE_HALFS_256); // if ax < bx, invert = -0.5
		invertMultX = _mm256_or_ps(_mm256_andnot_ps(aLessX, ONE_HALFS_256), invertMultX); // if ax >= bx, invert = 0.5

		auto invertMultY = _mm256_and_ps(aLessY, NEGATIVE_ONE_HALFS_256); // if ay < by, invert = -0.5
		invertMultY = _mm256_or_ps(_mm256_andnot_ps(aLessY, ONE_HALFS_256), invertMultY); // if ay >= by, invert = 0.5

		ax = _mm256_fmadd_ps(invertMultX, ddx, ax);
		ay = _mm256_fmadd_ps(invertMultY, ddy, ay);
		bx = _mm256_fnmadd_ps(invertMultX, ddx, bx);
		by = _mm256_fnmadd_ps(invertMultY, ddy, by);
		// rectify positions done now

		auto invTanLen = _mm256_rsqrt_ps(_mm256_add_ps(dx2, dy2));
		dxOrig = _mm256_mul_ps(dxOrig, invTanLen); // normalize
		dyOrig = _mm256_mul_ps(dyOrig, invTanLen); // normalize

		auto avAlongTan = _mm256_fnmadd_ps(avy, dxOrig, _mm256_mul_ps(avx, dyOrig)); // avy*-dx + avx*dy
		auto avAlongTanX = _mm256_mul_ps(avAlongTan, dyOrig);
		auto avAlongPerpX = _mm256_sub_ps(avx, avAlongTanX);
		auto negAvAlongTanY = _mm256_mul_ps(avAlongTan, dxOrig);
		auto avAlongPerpY = _mm256_add_ps(avy, negAvAlongTanY);

		auto bvAlongTan = _mm256_fnmadd_ps(bvy, dxOrig, _mm256_mul_ps(bvx, dyOrig)); // bvy*-dx + bvx*dy
		auto bvAlongTanX = _mm256_mul_ps(bvAlongTan, dyOrig);
		auto bvAlongPerpX = _mm256_sub_ps(bvx, bvAlongTanX);
		auto negBvAlongTanY = _mm256_mul_ps(bvAlongTan, dxOrig);
		auto bvAlongPerpY = _mm256_add_ps(bvy, negBvAlongTanY);

		avx = _mm256_add_ps(avAlongTanX, bvAlongPerpX);
		avy = _mm256_sub_ps(bvAlongPerpY, negAvAlongTanY);
		bvx = _mm256_add_ps(bvAlongTanX, avAlongPerpX);
		bvy = _mm256_sub_ps(avAlongPerpY, negBvAlongTanY);

		avx = _mm256_mul_ps(avx, ENERGY_LOSS_256);
		avy = _mm256_mul_ps(avy, ENERGY_LOSS_256);
		bvx = _mm256_mul_ps(bvx, ENERGY_LOSS_256);
		bvy = _mm256_mul_ps(bvy, ENERGY_LOSS_256);

		auto ax_ = (GLfloat*)&ax;
		auto ay_ = (GLfloat*)&ay;
		auto avx_ = (GLfloat*)&avx;
		auto avy_ = (GLfloat*)&avy;
		auto bx_ = (GLfloat*)&bx;
		auto by_ = (GLfloat*)&by;
		auto bvx_ = (GLfloat*)&bvx;
		auto bvy_ = (GLfloat*)&bvy;

		for (int k = 0; k < remaining; k++) {
			this->x[pairList[n+k].i] = ax_[k];
			this->y[pairList[n+k].i] = ay_[k];
			// this->vx[pairList[n+k].i] = avx_[k];
			// this->vy[pairList[n+k].i] = avy_[k];
			this->x[pairList[n+k].j] = bx_[k];
			this->y[pairList[n+k].j] = by_[k];
			// this->vx[pairList[n+k].j] = bvx_[k];
			// this->vy[pairList[n+k].j] = bvy_[k];
		}

	}
}