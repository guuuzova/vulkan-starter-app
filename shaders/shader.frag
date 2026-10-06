#version 450 core

layout(location = 0) in vec3 in_color;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform ObjectUniforms {
	mat4 matrix;
	vec4 color;
	float use_vertex_colors;
} object_uniforms;

void main() {
	// mix(a, b, k) = a * (1 - k) + b * k: при k = 0 берём белый, при k = 1 - цвет вершины
	const vec3 vertex_color = mix(vec3(1), in_color, object_uniforms.use_vertex_colors);

	out_color = vec4(vertex_color * object_uniforms.color.rgb, 1);
}
