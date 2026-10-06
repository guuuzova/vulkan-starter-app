#pragma once

#include <cmath>

namespace math {

struct Vector3 {
	float x, y, z;
};

// Матрица 4x4. Элементы хранятся по столбцам (column-major), как mat4 в GLSL,
// поэтому её можно копировать в uniform буфер без перестановки элементов
struct Matrix4 {
	float elements[16];

	// Доступ к элементу по номеру строки и столбца
	float& at(int row, int column) { return elements[column * 4 + row]; }
	float at(int row, int column) const { return elements[column * 4 + row]; }

	static Matrix4 zero() {
		Matrix4 result;
		for (int i = 0; i < 16; ++i) {
			result.elements[i] = 0.0f;
		}
		return result;
	}

	static Matrix4 identity() {
		Matrix4 result = zero();
		for (int i = 0; i < 4; ++i) {
			result.at(i, i) = 1.0f;
		}
		return result;
	}

	// Перемещение: вектор сдвига в последнем столбце (однородные координаты)
	static Matrix4 translation(Vector3 offset) {
		Matrix4 result = identity();
		result.at(0, 3) = offset.x;
		result.at(1, 3) = offset.y;
		result.at(2, 3) = offset.z;
		return result;
	}

	// Растяжение вдоль осей
	static Matrix4 scale(Vector3 factor) {
		Matrix4 result = identity();
		result.at(0, 0) = factor.x;
		result.at(1, 1) = factor.y;
		result.at(2, 2) = factor.z;
		return result;
	}

	// Поворот вокруг оси X на угол в радианах
	static Matrix4 rotationX(float angle) {
		const float c = std::cos(angle);
		const float s = std::sin(angle);
		Matrix4 result = identity();
		result.at(1, 1) = c;
		result.at(1, 2) = -s;
		result.at(2, 1) = s;
		result.at(2, 2) = c;
		return result;
	}

	// Поворот вокруг оси Y на угол в радианах
	static Matrix4 rotationY(float angle) {
		const float c = std::cos(angle);
		const float s = std::sin(angle);
		Matrix4 result = identity();
		result.at(0, 0) = c;
		result.at(0, 2) = s;
		result.at(2, 0) = -s;
		result.at(2, 2) = c;
		return result;
	}

	// Поворот вокруг оси Z на угол в радианах
	static Matrix4 rotationZ(float angle) {
		const float c = std::cos(angle);
		const float s = std::sin(angle);
		Matrix4 result = identity();
		result.at(0, 0) = c;
		result.at(0, 1) = -s;
		result.at(1, 0) = s;
		result.at(1, 1) = c;
		return result;
	}

	// Матрица перспективной проекции (лекция 1, слайд 29)
	// fov - вертикальный угол обзора в радианах, aspect - отношение ширины к высоте,
	// z_near и z_far - расстояния до ближней и дальней плоскостей отсечения
	static Matrix4 perspective(float fov, float aspect, float z_near, float z_far) {
		const float t = std::tan(fov / 2.0f);
		Matrix4 result = zero();
		result.at(0, 0) = 1.0f / (aspect * t);
		result.at(1, 1) = 1.0f / t;
		result.at(2, 2) = z_far / (z_far - z_near);
		result.at(2, 3) = -z_far * z_near / (z_far - z_near);
		result.at(3, 2) = 1.0f;
		return result;
	}

	// Матрица ортографической проекции (лекция 1, слайд 25): композиция перемещения
	// центра объёма вида в начало координат и растяжения его до канонического объёма
	// x: [left, right] -> [-1, 1], y: [top, bottom] -> [-1, 1], z: [z_near, z_far] -> [0, 1]
	static Matrix4 orthographic(float left, float right, float top, float bottom,
	                            float z_near, float z_far) {
		Matrix4 result = identity();
		result.at(0, 0) = 2.0f / (right - left);
		result.at(1, 1) = 2.0f / (bottom - top);
		result.at(2, 2) = 1.0f / (z_far - z_near);
		result.at(0, 3) = -(right + left) / (right - left);
		result.at(1, 3) = -(bottom + top) / (bottom - top);
		result.at(2, 3) = -z_near / (z_far - z_near);
		return result;
	}

	// Произведение матриц: сначала применяется правая, потом левая
	Matrix4 operator*(const Matrix4& other) const {
		Matrix4 result = zero();
		for (int row = 0; row < 4; ++row) {
			for (int column = 0; column < 4; ++column) {
				for (int k = 0; k < 4; ++k) {
					result.at(row, column) += at(row, k) * other.at(k, column);
				}
			}
		}
		return result;
	}
};

} // namespace math
