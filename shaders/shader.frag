#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

struct PointLight {
	vec3 position;
	vec3 color; // цвет, умноженный на интенсивность
};

struct SpotLight {
	vec3 position;
	float inner_cutoff; // косинус внутреннего угла
	vec3 color;         // цвет, умноженный на интенсивность
	float outer_cutoff; // косинус внешнего угла
	vec3 direction;
};

// Набор 0: общие данные сцены
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 directional_light_direction;
	vec3 directional_light_color;
	vec3 ambient_light;
	uint point_light_count;
	uint spot_light_count;
} scene_uniforms;

// Массивы источников света в storage буферах: длина массива в шейдере не задаётся
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

// Набор 1: данные модели
layout(std140, set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	vec3 albedo;
	vec3 specular;
	float alpha;
	float use_vertex_colors;
} model_uniforms;

// Модель Блинн-Фонга для одного источника света:
// L = (albedo + specular * (n * h)^alpha) * (n * l) * E
// n - нормаль, v - направление на камеру, l - направление на источник, light - свет E
vec3 blinnPhong(vec3 albedo, vec3 n, vec3 v, vec3 l, vec3 light) {
	const vec3 h = normalize(l + v); // половинчатый вектор между l и v

	const float n_dot_l = max(dot(n, l), 0);
	const float n_dot_h = max(dot(n, h), 0);

	return (albedo + model_uniforms.specular * pow(n_dot_h, model_uniforms.alpha)) *
	       n_dot_l * light;
}

void main() {
	const vec3 n = normalize(in_normal);
	const vec3 v = normalize(scene_uniforms.camera_position - in_world_position);

	// Альбедо материала, умноженное на цвет вершины (если цвета вершин включены)
	const vec3 albedo = model_uniforms.albedo *
	                    mix(vec3(1), in_color, model_uniforms.use_vertex_colors);

	// Рассеянный свет
	vec3 color = scene_uniforms.ambient_light * albedo;

	// Направленный источник: у всех точек одно направление, затухания нет
	color += blinnPhong(albedo, n, v, normalize(-scene_uniforms.directional_light_direction),
	                    scene_uniforms.directional_light_color);

	// Точечные источники: затухание по закону обратных квадратов 1 / D^2
	for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
		const vec3 to_light = point_lights[i].position - in_world_position;
		const float light_distance = length(to_light);
		const float falloff = 1.0 / (light_distance * light_distance);

		color += blinnPhong(albedo, n, v, to_light / light_distance,
		                    point_lights[i].color * falloff);
	}

	// Прожекторные источники: затухание 1 / D^2 и мягкие края между внутренним
	// и внешним углом
	for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
		const vec3 to_light = spot_lights[i].position - in_world_position;
		const float light_distance = length(to_light);
		const vec3 l = to_light / light_distance;
		const float falloff = 1.0 / (light_distance * light_distance);

		// Косинус угла между осью прожектора и направлением на точку:
		// внутри внутреннего угла edge = 1, за внешним углом edge = 0, между ними
		// плавный переход
		const float theta = dot(-l, normalize(spot_lights[i].direction));
		const float epsilon = max(spot_lights[i].inner_cutoff - spot_lights[i].outer_cutoff, 1e-4);
		const float edge = clamp((theta - spot_lights[i].outer_cutoff) / epsilon, 0, 1);

		color += blinnPhong(albedo, n, v, l, spot_lights[i].color * falloff * edge);
	}

	out_color = vec4(color, 1);
}
