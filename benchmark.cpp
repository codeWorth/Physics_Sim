#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace std::chrono;

void t_load(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_loadu_ps(floats[(i*3) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_loadu_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_permute(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	int32_t a[8] = {7, 0, 1, 2, 3, 4, 5, 6};
	__m256i permute = _mm256_loadu_si256((__m256i*) &a);

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_permutevar8x32_ps(groups[(i*3) % 100], permute);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_permutevar8x32_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_sub(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_sub_ps(groups[(i*9) % 100], groups[(i*7) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_sub_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_mul(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_mul_ps(groups[(i*9) % 100], groups[(i*7) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_mul_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_cmp(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_cmp_ps(groups[(i*9) % 100], groups[(i*7) % 100], _CMP_GE_OQ);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_cmp_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_and(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_and_ps(groups[(i*9) % 100], groups[(i*7) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_and_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_rsqrt(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_rsqrt_ps(groups[(i*7) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_rsqrt_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_rcp(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_rcp_ps(groups[(i*7) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_rcp_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_shift(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_slli_epi32(*(__m256i*)&groups[(i*7) % 100], 31);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_slli_epi32 - nanoseconds per avg: " << dtPer << endl;

}

void t_fmadd(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		_mm256_fmadd_ps(groups[(i*3) % 100], groups[(i*7) % 100], groups[(i*5) % 100]);
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "_mm256_fmadd_ps - nanoseconds per avg: " << dtPer << endl;

}

void t_floor(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	__m256 groups[100];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
		groups[i] = _mm256_loadu_ps(floats[i]);
	}

	int32_t b[8] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
	auto WINDOW_WIDTH_256 = _mm256_loadu_si256((__m256i*) &b);
	
	uint8_t BUFFER[13000]; // should be large enough to not cause segfault

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		for (int j = 0; j < 1400; j += 8) {
			auto xi = _mm256_cvttps_epi32(groups[(i*3+j*2) % 100]);
			auto yi = _mm256_cvttps_epi32(groups[(i*7+j*3) % 100]);
			xi = _mm256_srli_epi32(xi, 3); // divide by PHYSICS_SCALE
			yi = _mm256_srli_epi32(yi, 3); // divide by PHYSICS_SCALE
			yi = _mm256_mul_epi32(yi, WINDOW_WIDTH_256); // y*WINDOW_WIDTH
			yi = _mm256_add_epi32(yi, xi); // y*WINDOW_WIDTH + x

			int32_t* Is = (int32_t*) &yi;
			for (int j = 0; j < 8; j++) {
				BUFFER[Is[j]] = 255;
			}
		}
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "floor replacement - nanoseconds per avg: " << dtPer << endl;

}

void t_floorSlow(long count) {

	high_resolution_clock timer;
	float floats[100][8];
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 8; j++) {
			floats[i][j] = (double)((i+j*3) % 1000) / 85.0f;
		}
	}

	auto WINDOW_WIDTH = 1000;
	uint8_t BUFFER[13000]; // should be large enough to not cause segfault

	auto start = timer.now();
	for (long i = 0; i < count; i++) {
		for (int j = 0; j < 1400; j++) {
			int x = static_cast<int>(std::floor(floats[	(i*3 + (j>>3)*5)  %  100	][	j%8	])) >> 3;
			int y = static_cast<int>(std::floor(floats[	(i*3 + (j>>3)*3)  %  100	][	j%8	])) >> 3;
			BUFFER[y*WINDOW_WIDTH + x] = 255;
		}
	}
	auto stop = timer.now();
	
	long long dt = duration_cast<nanoseconds>(stop - start).count();
	double dtPer = (double)dt / count;

	cout << "floor original - nanoseconds per avg: " << dtPer << endl;

}
 

int main(void) {

	const long count = 100000000;

	t_load(count);
	t_permute(count);
	t_sub(count);
	t_mul(count);
	t_cmp(count);
	t_and(count);
	t_rsqrt(count);
	t_rcp(count);
	t_shift(count);
	t_fmadd(count);
	t_floor(1000);
	t_floorSlow(1000);

	return 0;
}