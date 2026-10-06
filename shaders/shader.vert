#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform ObjectUniforms {
	mat4 matrix;
	vec4 color;              // цвет фигуры из интерфейса
	float use_vertex_colors; // 1 - умножать на цвета вершин, 0 - не использовать их
} object_uniforms;

void main() {
	gl_Position = object_uniforms.matrix * vec4(in_position, 1);
	out_color = in_color;
}
