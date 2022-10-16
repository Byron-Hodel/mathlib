#ifndef MATH_H
#define MATH_H

#include <stdint.h>

// TODO: Add support for msvc
#define MATH_STRUCT_ALIGN(a) __attribute__((aligned(a)))

typedef struct MATH_STRUCT_ALIGN(16) {
	union {
		struct {
			float x, y;
		};
		float value[2];
	};
} vec2_t;

typedef struct MATH_STRUCT_ALIGN(16) {
	union {
		struct {
			double x, y;
		};
		double value[2];
	};
} vec2d_t;


typedef struct MATH_STRUCT_ALIGN(16) {
	union {
		struct {
			float x, y, z;
		};
		float value[3];
	};
} vec3_t;

typedef struct MATH_STRUCT_ALIGN(32) {
	union {
		struct {
			double x, y, z;
		};
		double value[3];
	};
} vec3d_t;



typedef struct MATH_STRUCT_ALIGN(16) {
	union {
		struct {
			float x, y, z, w;
		};
		float value[4];
	};
} vec4_t;

typedef struct MATH_STRUCT_ALIGN(32) {
	union {
		struct {
			double x, y, z, w;
		};
		double value[4];
	};
} vec4d_t;

typedef struct {
	vec2_t rows[2];
} mat2x2_t;

typedef struct {
	vec3_t rows[3];
} mat3x3_t;

typedef struct {
	vec4_t rows[4];
} mat4x4_t;

// TODO: more matrices types and matrices that use doubles 

inline vec2_t vec2_set(const float x, const float y);
inline vec2d_t vec2d_set(const double x, const double y);
inline vec3_t vec3_set(const float x, const float y, const float z);
inline vec3d_t vec3d_set(const double x, const double y, const double z);
inline vec4_t vec4_set(const float x, const float y, const float z, const float w);
inline vec4d_t vec4d_set(const double x, const double y, const double z, const double w);

inline vec2_t vec2_add(const vec2_t a, const vec2_t b);
inline vec2_t vec2_sub(const vec2_t a, const vec2_t b);
inline vec2_t vec2_mul(const vec2_t a, const vec2_t b);
inline vec2_t vec2_div(const vec2_t a, const vec2_t b);
inline float vec2_dot(const vec2_t a, const vec2_t b);

inline vec2d_t vec2d_add(const vec2d_t a, const vec2d_t b);
inline vec2d_t vec2d_sub(const vec2d_t a, const vec2d_t b);
inline vec2d_t vec2d_mul(const vec2d_t a, const vec2d_t b);
inline vec2d_t vec2d_div(const vec2d_t a, const vec2d_t b);
inline float vec2d_dot(const vec2d_t a, const vec2d_t b);



inline vec3_t vec3_add(const vec3_t a, const vec3_t b);
inline vec3_t vec3_sub(const vec3_t a, const vec3_t b);
inline vec3_t vec3_mul(const vec3_t a, const vec3_t b);
inline vec3_t vec3_div(const vec3_t a, const vec3_t b);
inline float vec3_dot(const vec3_t a, const vec3_t b);

inline vec3d_t vec3d_add(const vec3d_t a, const vec3d_t b);
inline vec3d_t vec3d_sub(const vec3d_t a, const vec3d_t b);
inline vec3d_t vec3d_mul(const vec3d_t a, const vec3d_t b);
inline vec3d_t vec3d_div(const vec3d_t a, const vec3d_t b);
inline float vec3d_dot(const vec3d_t a, const vec3d_t b);



inline vec4_t vec4_add(const vec4_t a, const vec4_t b);
inline vec4_t vec4_sub(const vec4_t a, const vec4_t b);
inline vec4_t vec4_mul(const vec4_t a, const vec4_t b);
inline vec4_t vec4_div(const vec4_t a, const vec4_t b);
inline float vec4_dot(const vec4_t a, const vec4_t b);

inline vec4d_t vec4d_add(const vec4d_t a, const vec4d_t b);
inline vec4d_t vec4d_sub(const vec4d_t a, const vec4d_t b);
inline vec4d_t vec4d_mul(const vec4d_t a, const vec4d_t b);
inline vec4d_t vec4d_div(const vec4d_t a, const vec4d_t b);
inline float vec4d_dot(const vec4d_t a, const vec4d_t b);


inline mat2x2_t mat2x2_add(const mat2x2_t a, const mat2x2_t b);
inline mat2x2_t mat2x2_sub(const mat2x2_t a, const mat2x2_t b);
inline mat2x2_t mat2x2_mul(const mat2x2_t a, const mat2x2_t b);

inline mat3x3_t mat3x3_add(const mat3x3_t a, const mat3x3_t b);
inline mat3x3_t mat3x3_sub(const mat3x3_t a, const mat3x3_t b);
inline mat3x3_t mat3x3_mul(const mat3x3_t a, const mat3x3_t b);

inline mat4x4_t mat4x4_add(const mat4x4_t a, const mat4x4_t b);
inline mat4x4_t mat4x4_sub(const mat4x4_t a, const mat4x4_t b);
inline mat4x4_t mat4x4_mul(const mat4x4_t a, const mat4x4_t b);


// TODO: add cross product function for vectors
// TODO: add function to multiply a matrix by a single value


#ifdef MATH_IMPLIMENTATION

vec2_t vec2_set(const float x, const float y) {
	vec2_t v;
	v.x = x;
	v.y = y;
	return v;
}
vec2d_t vec2d_set(const double x, const double y) {
	vec2d_t v;
	v.x = x;
	v.y = y;
	return v;
}
vec3_t vec3_set(const float x, const float y, const float z) {
	vec3_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}
vec3d_t vec3d_set(const double x, const double y, const double z) {
	vec3d_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	return v;
}
vec4_t vec4_set(const float x, const float y, const float z, const float w) {
	vec4_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}
vec4d_t vec4d_set(const double x, const double y, const double z, const double w) {
	vec4d_t v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}

// NOTE: Optimizer should be able to optimize dot product well enough
// TODO: maybe use simd for dot product using SSE 4.1 (_mm_dp_ps)

float vec2_dot(const vec2_t a, const vec2_t b) {
	return a.x * b.x + a.y * b.y;
}
float vec2d_dot(const vec2d_t a, const vec2d_t b) {
	return a.x * b.x + a.y * b.y;
}
float vec3_dot(const vec3_t a, const vec3_t b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
float vec3d_dot(const vec3d_t a, const vec3d_t b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
float vec4_dot(const vec4_t a, const vec4_t b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
float vec4d_dot(const vec4d_t a, const vec4d_t b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

mat2x2_t mat2x2_add(const mat2x2_t a, const mat2x2_t b) {
	mat2x2_t mat;
	mat.rows[0] = vec2_add(a.rows[0], b.rows[0]);
	mat.rows[1] = vec2_add(a.rows[1], b.rows[1]);
	return mat;
}
mat2x2_t mat2x2_sub(const mat2x2_t a, const mat2x2_t b) {
	mat2x2_t mat;
	mat.rows[0] = vec2_sub(a.rows[0], b.rows[0]);
	mat.rows[1] = vec2_sub(a.rows[1], b.rows[1]);
	return mat;
}
mat3x3_t mat3x3_add(const mat3x3_t a, const mat3x3_t b) {
	mat3x3_t mat;
	mat.rows[0] = vec3_add(a.rows[0], b.rows[0]);
	mat.rows[1] = vec3_add(a.rows[1], b.rows[1]);
	mat.rows[2] = vec3_add(a.rows[2], b.rows[2]);
	return mat;
}
mat3x3_t mat3x3_sub(const mat3x3_t a, const mat3x3_t b) {
	mat3x3_t mat;
	mat.rows[0] = vec3_sub(a.rows[0], b.rows[0]);
	mat.rows[1] = vec3_sub(a.rows[1], b.rows[1]);
	mat.rows[2] = vec3_sub(a.rows[2], b.rows[2]);
	return mat;
}
mat4x4_t mat4x4_add(const mat4x4_t a, const mat4x4_t b) {
	mat4x4_t mat;
	mat.rows[0] = vec4_add(a.rows[0], b.rows[0]);
	mat.rows[1] = vec4_add(a.rows[1], b.rows[1]);
	mat.rows[2] = vec4_add(a.rows[2], b.rows[2]);
	mat.rows[3] = vec4_add(a.rows[3], b.rows[3]);
	return mat;
}
mat4x4_t mat4x4_sub(const mat4x4_t a, const mat4x4_t b) {
	mat4x4_t mat;
	mat.rows[0] = vec4_sub(a.rows[0], b.rows[0]);
	mat.rows[1] = vec4_sub(a.rows[1], b.rows[1]);
	mat.rows[2] = vec4_sub(a.rows[2], b.rows[2]);
	mat.rows[3] = vec4_sub(a.rows[3], b.rows[3]);
	return mat;
}

#if __SSE2__

#include <emmintrin.h>

vec2_t vec2_add(const vec2_t a, const vec2_t b) {
	vec2_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_add_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec2_t vec2_sub(const vec2_t a, const vec2_t b) {
	vec2_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_sub_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec2_t vec2_mul(const vec2_t a, const vec2_t b) {
	vec2_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_mul_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec2_t vec2_div(const vec2_t a, const vec2_t b) {
	vec2_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_div_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}

vec2d_t vec2d_add(const vec2d_t a, const vec2d_t b) {
	vec2d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_add_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	return vec;
}
vec2d_t vec2d_sub(const vec2d_t a, const vec2d_t b) {
	vec2d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_sub_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	return vec;
}
vec2d_t vec2d_mul(const vec2d_t a, const vec2d_t b) {
	vec2d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_mul_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	return vec;
}
vec2d_t vec2d_div(const vec2d_t a, const vec2d_t b) {
	vec2d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_div_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	return vec;
}



vec3_t vec3_add(const vec3_t a, const vec3_t b) {
	vec3_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_add_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec3_t vec3_sub(const vec3_t a, const vec3_t b) {
	vec3_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_sub_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec3_t vec3_mul(const vec3_t a, const vec3_t b) {
	vec3_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_mul_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec3_t vec3_div(const vec3_t a, const vec3_t b) {
	vec3_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_div_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}

vec3d_t vec3d_add(const vec3d_t a, const vec3d_t b) {
	vec3d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_add_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	vec.value[2] = a.z + b.z;
	return vec;
}
vec3d_t vec3d_sub(const vec3d_t a, const vec3d_t b) {
	vec3d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_sub_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	vec.value[2] = a.z - b.z;
	return vec;
}
vec3d_t vec3d_mul(const vec3d_t a, const vec3d_t b) {
	vec3d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_mul_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	vec.value[2] = a.z * b.z;
	return vec;
}
vec3d_t vec3d_div(const vec3d_t a, const vec3d_t b) {
	vec3d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_div_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	vec.value[2] = a.z / b.z;
	return vec;
}



vec4_t vec4_add(const vec4_t a, const vec4_t b) {
	vec4_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_add_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec4_t vec4_sub(const vec4_t a, const vec4_t b) {
	vec4_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_sub_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec4_t vec4_mul(const vec4_t a, const vec4_t b) {
	vec4_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_mul_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}
vec4_t vec4_div(const vec4_t a, const vec4_t b) {
	vec4_t vec;
	__m128 _a = _mm_load_ps(a.value);
	__m128 _b = _mm_load_ps(b.value);
	__m128 r = _mm_div_ps(_a, _b);
	_mm_store_ps(vec.value, r);
	return vec;
}


vec4d_t vec4d_add(const vec4d_t a, const vec4d_t b) {
	vec4d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_add_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	_a = _mm_load_pd(a.value + 2);
	_b = _mm_load_pd(b.value + 2);
	r = _mm_add_pd(_a, _b);
	_mm_store_pd(vec.value+2, r);
	return vec;
}
vec4d_t vec4d_sub(const vec4d_t a, const vec4d_t b) {
	vec4d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_sub_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	_a = _mm_load_pd(a.value + 2);
	_b = _mm_load_pd(b.value + 2);
	r = _mm_sub_pd(_a, _b);
	_mm_store_pd(vec.value+2, r);
	return vec;
}
vec4d_t vec4d_mul(const vec4d_t a, const vec4d_t b) {
	vec4d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_mul_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	_a = _mm_load_pd(a.value + 2);
	_b = _mm_load_pd(b.value + 2);
	r = _mm_mul_pd(_a, _b);
	_mm_store_pd(vec.value+2, r);
	return vec;
}
vec4d_t vec4d_div(const vec4d_t a, const vec4d_t b) {
	vec4d_t vec;
	__m128d _a = _mm_load_pd(a.value);
	__m128d _b = _mm_load_pd(b.value);
	__m128d r = _mm_div_pd(_a, _b);
	_mm_store_pd(vec.value, r);
	_a = _mm_load_pd(a.value + 2);
	_b = _mm_load_pd(b.value + 2);
	r = _mm_div_pd(_a, _b);
	_mm_store_pd(vec.value+2, r);
	return vec;
}


mat2x2_t mat2x2_mul(const mat2x2_t a, const mat2x2_t b) {
	mat2x2_t mat;
	__m128 row_b0 = _mm_load_ps(b.rows[0].value);
	__m128 row_b1 = _mm_load_ps(b.rows[1].value);

	for(int i = 0; i < 2; i++) {
		__m128 multiplier_0 = _mm_set1_ps(a.rows[i].value[0]);
		__m128 multiplier_1 = _mm_set1_ps(a.rows[i].value[1]);

		__m128 mat_row = _mm_mul_ps(row_b0, multiplier_0);
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b1, multiplier_1));
		_mm_store_ps(mat.rows[i].value, mat_row);
	}

	return mat;
}

mat3x3_t mat3x3_mul(const mat3x3_t a, const mat3x3_t b) {
	mat3x3_t mat;
	__m128 row_b0 = _mm_load_ps(b.rows[0].value);
	__m128 row_b1 = _mm_load_ps(b.rows[1].value);
	__m128 row_b2 = _mm_load_ps(b.rows[2].value);

	for(int i = 0; i < 3; i++) {
		__m128 multiplier_0 = _mm_set1_ps(a.rows[i].value[0]);
		__m128 multiplier_1 = _mm_set1_ps(a.rows[i].value[1]);
		__m128 multiplier_2 = _mm_set1_ps(a.rows[i].value[2]);

		__m128 mat_row = _mm_mul_ps(row_b0, multiplier_0);
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b1, multiplier_1));
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b2, multiplier_2));
		_mm_store_ps(mat.rows[i].value, mat_row);
	}

	return mat;
}

mat4x4_t mat4x4_mul(const mat4x4_t a, const mat4x4_t b) {
	mat4x4_t mat;
	__m128 row_b0 = _mm_load_ps(b.rows[0].value);
	__m128 row_b1 = _mm_load_ps(b.rows[1].value);
	__m128 row_b2 = _mm_load_ps(b.rows[2].value);
	__m128 row_b3 = _mm_load_ps(b.rows[3].value);

	for(int i = 0; i < 4; i++) {
		__m128 multiplier_0 = _mm_set1_ps(a.rows[i].value[0]);
		__m128 multiplier_1 = _mm_set1_ps(a.rows[i].value[1]);
		__m128 multiplier_2 = _mm_set1_ps(a.rows[i].value[2]);
		__m128 multiplier_3 = _mm_set1_ps(a.rows[i].value[3]);

		__m128 mat_row = _mm_mul_ps(row_b0, multiplier_0);
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b1, multiplier_1));
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b2, multiplier_2));
		mat_row = _mm_add_ps(mat_row, _mm_mul_ps(row_b3, multiplier_3));
		_mm_store_ps(mat.rows[i].value, mat_row);
	}

	return mat;
}


#endif // __SSE2__

// TODO: add support for avx, excpecially for vec3d and vec4d
// NOTE: Not all functions should be avx, for example vec2, 3, and 4 operations dont need AVX

#endif // MATH_IMPLIMENTATION


#endif // MATH_H