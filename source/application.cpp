#include "application.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "math.hpp"

namespace application {

namespace {

struct Vertex {
	float position[3];
	float normal[3];
	float color[3];
};

// Треугольная пирамида: вершина сверху (ось y направлена вниз) и основание
// из правильного треугольника в плоскости y = 0.5
const math::Vector3 corners[] = {
	{ +0.000f, -1.0f, +0.0f }, // 0: вершина пирамиды
	{ +0.000f, +0.5f, +1.0f }, // 1: основание, угол 90°
	{ -0.866f, +0.5f, -0.5f }, // 2: основание, угол 210°
	{ +0.866f, +0.5f, -0.5f }, // 3: основание, угол 330°
};

// 4 грани по 3 угла, обход по часовой стрелке, если смотреть снаружи
const uint32_t faces[4][3] = {
	{ 0, 2, 1 }, // боковая грань
	{ 0, 3, 2 }, // боковая грань
	{ 0, 1, 3 }, // боковая грань
	{ 1, 2, 3 }, // основание
};

// У каждой грани свои 3 вершины, потому что нормаль у вершины своя для каждой грани.
// Вершины и индексы заполняются в buildPyramid()
constexpr size_t vertex_count = 4 * 3;
Vertex vertices[vertex_count];
uint32_t indices[vertex_count];

// Общие данные сцены (зеркалит SceneUniforms в шейдере)
// По правилам std140 vec3 выравнивается по 16 байтам, поэтому после каждого vec3
// стоит отступ, а point_light_count занимает 4 свободных байта после ambient_light
struct SceneUniforms {
	float view_projection[16];            // байты 0-63
	float camera_position[3];             // 64-75
	float padding0;
	float directional_light_direction[3]; // 80-91
	float padding1;
	float directional_light_color[3];     // 96-107
	float padding2;
	float ambient_light[3];               // 112-123
	uint32_t point_light_count;           // 124-127
	uint32_t spot_light_count;            // 128-131
};

// Элементы массивов в storage буферах (зеркалят PointLight и SpotLight в шейдере)
// По правилам std430 vec3 тоже выравнивается по 16 байтам
struct PointLightData {
	float position[3]; // 0-11
	float padding0;
	float color[3];    // 16-27
	float padding1;
};

struct SpotLightData {
	float position[3];  // 0-11
	float inner_cutoff; // 12-15
	float color[3];     // 16-27
	float outer_cutoff; // 28-31
	float direction[3]; // 32-43
	float padding;
};

// Данные одной модели (зеркалит ModelUniforms в шейдере, std140)
struct ModelUniforms {
	float model[16];         // 0-63
	float albedo[3];         // 64-75
	float padding0;
	float specular[3];       // 80-91
	float alpha;             // 92-95
	float use_vertex_colors; // 96-99
};

// Один объект на сцене: его параметры из интерфейса и его собственные ресурсы Vulkan
struct Object {
	int id; // постоянный номер объекта для интерфейса

	// Трансформация
	float position[3];
	float rotation[3]; // углы в градусах
	float scale[3];

	// Материал по модели Блинн-Фонга
	float albedo[3];        // цвет диффузной составляющей
	float specular[3];      // цвет блика
	float alpha;            // коэффициент блика (чем больше, тем меньше и резче блик)
	bool use_vertex_colors; // умножать альбедо на цвета вершин

	// Анимация: движение по восьмёрке вокруг position и вращение
	bool animation_playing;
	float animation_time;    // "время" анимации, идёт только при воспроизведении
	float animation_speed;   // множитель скорости
	float trajectory_radius; // радиус траектории
	float spin_speed;        // скорость вращения вокруг своей оси, градусов в секунду
	float swing_amplitude;   // амплитуда раскачивания вокруг оси Z, градусы

	// У каждого объекта свой uniform буфер и свой набор дескрипторов (set = 1)
	VkBuffer uniform_buffer;
	VmaAllocation uniform_buffer_allocation;
	ModelUniforms* uniform_buffer_memory;
	VkDescriptorSet descriptor_set;
};

// Источники света в интерфейсе
struct PointLight {
	int id;
	float position[3];
	float color[3];
	float intensity;
};

struct SpotLight {
	int id;
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float inner_angle; // градусы
	float outer_angle; // градусы
};

// Сколько объектов максимум можно создать: столько наборов set = 1 выделяет пул
constexpr uint32_t max_objects = 16;

// Сколько источников помещается в storage буферы. Это ограничение только
// размера буфера на стороне процессора, в шейдере длина массивов не задаётся
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;

std::vector<Object> objects;
int next_object_id = 1;

std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;
int next_light_id = 1;

// Описания наборов дескрипторов: set = 0 для сцены, set = 1 для модели
VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;
VkDescriptorPool vk_descriptor_pool;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

VkBuffer vk_vertex_buffer;
VmaAllocation vk_vertex_buffer_allocation;
Vertex* vk_vertex_buffer_memory;

VkBuffer vk_index_buffer;
VmaAllocation vk_index_buffer_allocation;
uint32_t* vk_index_buffer_memory;

// Ресурсы сцены: uniform буфер, два storage буфера и набор дескрипторов на них
VkBuffer vk_scene_uniform_buffer;
VmaAllocation vk_scene_uniform_buffer_allocation;
SceneUniforms* vk_scene_uniform_buffer_memory;

VkBuffer vk_point_light_buffer;
VmaAllocation vk_point_light_buffer_allocation;
PointLightData* vk_point_light_buffer_memory;

VkBuffer vk_spot_light_buffer;
VmaAllocation vk_spot_light_buffer_allocation;
SpotLightData* vk_spot_light_buffer_memory;

VkDescriptorSet vk_scene_descriptor_set;

// Камера
float camera_position[3] = { 0.0f, 0.0f, 0.0f };
float camera_pitch = 0.0f; // поворот вокруг оси X в градусах
float camera_yaw = 0.0f;   // поворот вокруг оси Y в градусах

// Параметры проекции камеры
bool use_perspective = true;
float perspective_fov = 60.0f;  // вертикальный угол обзора в градусах
float orthographic_size = 4.0f; // половина высоты видимой области

// Рассеянный свет
float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.1f;

// Направленный источник света
float directional_light_direction[3] = { 0.5f, 1.0f, 1.0f };
float directional_light_color[3] = { 1.0f, 1.0f, 1.0f };
float directional_light_intensity = 0.5f;

double previous_time = 0.0;

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// Процедурно строит сетку пирамиды: для каждой грани 3 вершины с нормалью грани
void buildPyramid() {
	for (size_t face = 0; face < 4; ++face) {
		const math::Vector3 a = corners[faces[face][0]];
		const math::Vector3 b = corners[faces[face][1]];
		const math::Vector3 c = corners[faces[face][2]];

		// Векторное произведение двух рёбер грани перпендикулярно грани.
		// Порядок множителей выбран так, чтобы нормаль смотрела наружу пирамиды
		const math::Vector3 normal = math::normalize(math::cross(c - a, b - a));

		for (size_t k = 0; k < 3; ++k) {
			const math::Vector3 corner = corners[faces[face][k]];
			Vertex& vertex = vertices[face * 3 + k];

			vertex.position[0] = corner.x;
			vertex.position[1] = corner.y;
			vertex.position[2] = corner.z;

			vertex.normal[0] = normal.x;
			vertex.normal[1] = normal.y;
			vertex.normal[2] = normal.z;

			indices[face * 3 + k] = uint32_t(face * 3 + k);
		}
	}
}

// Процедурные цвета вершин: каждая координата вершины в локальной системе
// переводится в диапазон [0, 1] внутри габаритов фигуры и становится каналом цвета
// (x -> красный, y -> зелёный, z -> синий)
void assignVertexColors() {
	float min[3] = { vertices[0].position[0], vertices[0].position[1], vertices[0].position[2] };
	float max[3] = { min[0], min[1], min[2] };

	// Ищем габариты фигуры
	for (size_t i = 1; i < vertex_count; ++i) {
		for (int axis = 0; axis < 3; ++axis) {
			min[axis] = std::min(min[axis], vertices[i].position[axis]);
			max[axis] = std::max(max[axis], vertices[i].position[axis]);
		}
	}

	for (size_t i = 0; i < vertex_count; ++i) {
		for (int axis = 0; axis < 3; ++axis) {
			vertices[i].color[axis] = (vertices[i].position[axis] - min[axis]) /
			                          (max[axis] - min[axis]);
		}
	}
}

// Объект с настройками по умолчанию в точке (x, y, z) с альбедо (r, g, b)
// Поля Vulkan здесь не заполняются: их создаёт createObjectResources()
Object makeObject(float x, float y, float z, float r, float g, float b) {
	Object object = {};

	object.position[0] = x;
	object.position[1] = y;
	object.position[2] = z;

	object.rotation[0] = 20.0f;
	object.rotation[1] = 30.0f;
	object.rotation[2] = 0.0f;

	object.scale[0] = 1.0f;
	object.scale[1] = 1.0f;
	object.scale[2] = 1.0f;

	object.albedo[0] = r;
	object.albedo[1] = g;
	object.albedo[2] = b;

	object.specular[0] = 0.5f;
	object.specular[1] = 0.5f;
	object.specular[2] = 0.5f;

	object.alpha = 32.0f;
	object.use_vertex_colors = false;

	object.animation_playing = false;
	object.animation_time = 0.0f;
	object.animation_speed = 1.0f;
	object.trajectory_radius = 1.0f;
	object.spin_speed = 90.0f;
	object.swing_amplitude = 20.0f;

	return object;
}

// Создаёт буфер в памяти, доступной процессору для записи, и сразу отображает
// её (memory mapping), чтобы данные в буфер можно было копировать через memcpy
bool createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer& buffer, VmaAllocation& allocation, void** memory) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate buffer\n";
		return false;
	}

	if (vmaMapMemory(context.allocator, allocation, memory) != VK_SUCCESS) {
		std::cerr << "Failed to map buffer memory\n";
		return false;
	}

	return true;
}

void destroyMappedBuffer(VkBuffer buffer, VmaAllocation allocation) {
	auto& context = graphics::internal::context;

	vmaUnmapMemory(context.allocator, allocation);
	vmaDestroyBuffer(context.allocator, buffer, allocation);
}

// Читаем SPIR-V файл, скомпилированный glslc, и создаём из него шейдерный модуль
VkShaderModule loadShaderModule(const char path[]) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file " << path << '\n';
		return nullptr;
	}

	const size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));

	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(graphics::internal::context.device,
	                         &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vk_vertex_shader = loadShaderModule("shaders/shader.vert.spv");
	VkShaderModule vk_fragment_shader = loadShaderModule("shaders/shader.frag.spv");
	if (vk_vertex_shader == nullptr || vk_fragment_shader == nullptr) {
		std::cerr << "Failed to load shaders\n";
		return false;
	}

	// Стадии шейдеров: вершинный и фрагментный
	const VkPipelineShaderStageCreateInfo stage_infos[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_fragment_shader,
			.pName = "main",
		},
	};

	// Input assembler: из какого буфера брать вершины и как разложить их на атрибуты
	const VkVertexInputBindingDescription vertex_bindings[] = {
		{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		},
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo input_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = sizeof(vertex_bindings) / sizeof(vertex_bindings[0]),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	// Primitive assembler: рисуем треугольники
	const VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	// Одна область отрисовки и одна область отрезания, размеры зададим в render
	const VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Растеризатор
	const VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo sample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// Тест глубины
	const VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	// Смешение цветов: формулу не задаём, просто пишем все компоненты RGBA
	const VkPipelineColorBlendAttachmentState attachment_info{
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo blend_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &attachment_info,
	};

	// Динамические параметры: размеры viewport и scissor задаём при записи команд
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	// Данные извне: два набора дескрипторов, set = 0 для сцены и set = 1 для модели
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stage_infos) / sizeof(stage_infos[0]),
		.pStages = stage_infos,
		.pVertexInputState = &input_state_info,
		.pInputAssemblyState = &assembly_state_info,
		.pViewportState = &viewport_info,
		.pRasterizationState = &raster_info,
		.pMultisampleState = &sample_info,
		.pDepthStencilState = &depth_info,
		.pColorBlendState = &blend_info,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const bool success = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                               &pipeline_info, nullptr,
	                                               &vk_pipeline) == VK_SUCCESS;

	// Шейдерные модули нужны только на время создания конвейера
	vkDestroyShaderModule(context.device, vk_fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vk_vertex_shader, nullptr);

	if (!success) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

// Вершинный и индексный буферы пирамиды
bool createMeshBuffers() {
	buildPyramid();
	assignVertexColors();

	if (!createMappedBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                        vk_vertex_buffer, vk_vertex_buffer_allocation,
	                        (void**)&vk_vertex_buffer_memory)) {
		return false;
	}

	memcpy(vk_vertex_buffer_memory, vertices, sizeof(vertices));

	if (!createMappedBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                        vk_index_buffer, vk_index_buffer_allocation,
	                        (void**)&vk_index_buffer_memory)) {
		return false;
	}

	memcpy(vk_index_buffer_memory, indices, sizeof(indices));

	return true;
}

// Описания двух наборов дескрипторов
bool createDescriptorSetLayouts() {
	auto& context = graphics::internal::context;

	// set = 0: данные сцены
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			// binding = 0: uniform буфер SceneUniforms
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			// binding = 1: storage буфер с массивом точечных источников
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			// binding = 2: storage буфер с массивом прожекторных источников
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_set_layout, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	// set = 1: данные модели
	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{
			// binding = 0: uniform буфер ModelUniforms
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo model_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(model_bindings) / sizeof(model_bindings[0]),
		.pBindings = model_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_set_layout, nullptr,
	                                &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	return true;
}

// Пул, из которого выделяются набор сцены и наборы всех объектов
bool createDescriptorPool() {
	auto& context = graphics::internal::context;

	// Uniform буферы: 1 в наборе сцены + по 1 в наборе каждого объекта
	// Storage буферы: 2 в наборе сцены
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + max_objects,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		// Разрешаем возвращать в пул наборы удалённых объектов через vkFreeDescriptorSets
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = 1 + max_objects,
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	return true;
}

// Буферы сцены и набор дескрипторов set = 0, который на них указывает
bool createSceneResources() {
	auto& context = graphics::internal::context;

	if (!createMappedBuffer((sizeof(SceneUniforms) + 0xf) & ~0xf, // выравниваем до 16 байт
	                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                        vk_scene_uniform_buffer, vk_scene_uniform_buffer_allocation,
	                        (void**)&vk_scene_uniform_buffer_memory)) {
		return false;
	}

	if (!createMappedBuffer(sizeof(PointLightData) * max_point_lights,
	                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                        vk_point_light_buffer, vk_point_light_buffer_allocation,
	                        (void**)&vk_point_light_buffer_memory)) {
		return false;
	}

	if (!createMappedBuffer(sizeof(SpotLightData) * max_spot_lights,
	                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                        vk_spot_light_buffer, vk_spot_light_buffer_allocation,
	                        (void**)&vk_spot_light_buffer_memory)) {
		return false;
	}

	const VkDescriptorSetAllocateInfo descriptor_set = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &descriptor_set,
	                             &vk_scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	const VkDescriptorBufferInfo scene_uniform_buffer_descriptor = {
		.buffer = vk_scene_uniform_buffer,
		.offset = 0,
		.range = sizeof(SceneUniforms),
	};

	const VkDescriptorBufferInfo point_light_buffer_descriptor = {
		.buffer = vk_point_light_buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};

	const VkDescriptorBufferInfo spot_light_buffer_descriptor = {
		.buffer = vk_spot_light_buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};

	const VkWriteDescriptorSet descriptor_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_uniform_buffer_descriptor,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &point_light_buffer_descriptor,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_scene_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &spot_light_buffer_descriptor,
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(descriptor_writes) / sizeof(descriptor_writes[0]),
	                       descriptor_writes, 0, nullptr);

	return true;
}

// Создаёт для объекта uniform буфер и набор дескрипторов set = 1, который на него указывает
bool createObjectResources(Object& object) {
	auto& context = graphics::internal::context;

	if (!createMappedBuffer((sizeof(ModelUniforms) + 0xf) & ~0xf, // выравниваем до 16 байт
	                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                        object.uniform_buffer, object.uniform_buffer_allocation,
	                        (void**)&object.uniform_buffer_memory)) {
		return false;
	}

	const VkDescriptorSetAllocateInfo descriptor_set = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_model_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &descriptor_set,
	                             &object.descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	const VkDescriptorBufferInfo uniform_buffer_descriptor = {
		.buffer = object.uniform_buffer,
		.offset = 0,
		.range = sizeof(ModelUniforms),
	};

	const VkWriteDescriptorSet descriptor_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &uniform_buffer_descriptor,
		},
	};

	vkUpdateDescriptorSets(context.device, sizeof(descriptor_writes) / sizeof(descriptor_writes[0]),
	                       descriptor_writes, 0, nullptr);

	return true;
}

// Возвращает набор дескрипторов объекта в пул и уничтожает его uniform буфер
void destroyObjectResources(Object& object) {
	auto& context = graphics::internal::context;

	vkFreeDescriptorSets(context.device, vk_descriptor_pool, 1, &object.descriptor_set);
	destroyMappedBuffer(object.uniform_buffer, object.uniform_buffer_allocation);
}

// Добавляет объект на сцену, если ещё есть место в пуле дескрипторов
bool addObject(Object object) {
	if (objects.size() >= max_objects) {
		return false;
	}

	if (!createObjectResources(object)) {
		return false;
	}

	object.id = next_object_id++;
	objects.push_back(object);
	return true;
}

// Удаляет объект со сцены
void removeObject(size_t index) {
	// GPU может ещё рисовать прошлый кадр с этим буфером и набором,
	// поэтому ждём, пока очередь опустеет, и только потом уничтожаем
	vkQueueWaitIdle(graphics::internal::context.graphics_queue);

	destroyObjectResources(objects[index]);
	objects.erase(objects.begin() + index);
}

void addPointLight(float x, float y, float z, float r, float g, float b, float intensity) {
	if (point_lights.size() < max_point_lights) {
		point_lights.push_back({ next_light_id++, { x, y, z }, { r, g, b }, intensity });
	}
}

void addSpotLight(float x, float y, float z, float dx, float dy, float dz,
                  float r, float g, float b, float intensity,
                  float inner_angle, float outer_angle) {
	if (spot_lights.size() < max_spot_lights) {
		spot_lights.push_back({ next_light_id++, { x, y, z }, { dx, dy, dz }, { r, g, b },
		                        intensity, inner_angle, outer_angle });
	}
}

// Интерфейс настроек одного объекта
// Возвращает false, если пользователь нажал "Remove"
bool objectInterface(Object& object) {
	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", object.position, 0.01f);
	ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
	ImGui::DragFloat3("Scale", object.scale, 0.01f, 0.1f, 5.0f);

	ImGui::SeparatorText("Material");
	ImGui::ColorEdit3("Albedo", object.albedo);
	ImGui::ColorEdit3("Specular", object.specular);
	ImGui::SliderFloat("Shininess", &object.alpha, 1.0f, 256.0f, "%.0f",
	                   ImGuiSliderFlags_Logarithmic);
	ImGui::Checkbox("Vertex colors", &object.use_vertex_colors);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(object.animation_playing ? "Pause" : "Play")) {
		object.animation_playing = !object.animation_playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		object.animation_time = 0.0f;
	}

	ImGui::SliderFloat("Speed", &object.animation_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &object.trajectory_radius, 0.0f, 3.0f);
	ImGui::SliderFloat("Spin (deg/s)", &object.spin_speed, -360.0f, 360.0f);
	ImGui::SliderFloat("Swing (deg)", &object.swing_amplitude, 0.0f, 90.0f);

	ImGui::Separator();
	return !ImGui::Button("Remove");
}

// Интерфейс настроек точечного источника
// Возвращает false, если пользователь нажал "Remove"
bool pointLightInterface(PointLight& light) {
	ImGui::DragFloat3("Position", light.position, 0.01f);
	ImGui::ColorEdit3("Color", light.color);
	ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);

	return !ImGui::Button("Remove");
}

// Интерфейс настроек прожекторного источника
// Возвращает false, если пользователь нажал "Remove"
bool spotLightInterface(SpotLight& light) {
	ImGui::DragFloat3("Position", light.position, 0.01f);
	ImGui::DragFloat3("Direction", light.direction, 0.01f);
	ImGui::ColorEdit3("Color", light.color);
	ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
	ImGui::SliderFloat("Inner angle (deg)", &light.inner_angle, 0.0f, 89.0f);
	ImGui::SliderFloat("Outer angle (deg)", &light.outer_angle, 0.0f, 89.0f);

	// Внешний угол не может быть меньше внутреннего
	if (light.outer_angle < light.inner_angle) {
		light.outer_angle = light.inner_angle;
	}

	return !ImGui::Button("Remove");
}

// Матрица модели объекта с учётом анимации
math::Matrix4 objectModelMatrix(const Object& object) {
	// Смещение по траектории-восьмёрке (кривая Лиссажу) и углы анимации
	const float t = object.animation_time;
	const float r = object.trajectory_radius;
	const float offset_x = r * std::sin(t);
	const float offset_y = r * 0.5f * std::sin(2.0f * t);
	const float offset_z = r * 0.5f * (std::cos(t) - 1.0f); // при t = 0 смещение 0

	const float spin_angle = object.spin_speed * t;
	const float swing_angle = object.swing_amplitude * std::sin(2.0f * t);

	// Сначала растяжение, затем повороты вокруг осей, затем перемещение
	// (матрицы применяются справа налево)
	// К значениям из интерфейса добавляется смещение и поворот от анимации
	return math::Matrix4::translation({ object.position[0] + offset_x,
	                                    object.position[1] + offset_y,
	                                    object.position[2] + offset_z }) *
	       math::Matrix4::rotationZ(toRadians(object.rotation[2] + swing_angle)) *
	       math::Matrix4::rotationY(toRadians(object.rotation[1] + spin_angle)) *
	       math::Matrix4::rotationX(toRadians(object.rotation[0])) *
	       math::Matrix4::scale({ object.scale[0], object.scale[1], object.scale[2] });
}

// Матрица вида (лекция 4, слайды 7-10)
// Камера ставится в мир матрицей C = T(position) * Ry(yaw) * Rx(pitch).
// Матрица вида - обратная ей: V = C^-1 = Rx(-pitch) * Ry(-yaw) * T(-position),
// потому что (ABC)^-1 = C^-1 B^-1 A^-1, обратный поворот - поворот на минус угол,
// а обратное перемещение - перемещение на минус вектор
math::Matrix4 viewMatrix() {
	return math::Matrix4::rotationX(toRadians(-camera_pitch)) *
	       math::Matrix4::rotationY(toRadians(-camera_yaw)) *
	       math::Matrix4::translation({ -camera_position[0],
	                                    -camera_position[1],
	                                    -camera_position[2] });
}

} // namespace

bool initialize() {
	// Описания наборов дескрипторов нужны конвейеру, поэтому создаём их первыми
	if (!createDescriptorSetLayouts()) {
		return false;
	}

	if (!createPipeline()) {
		return false;
	}

	if (!createMeshBuffers()) {
		return false;
	}

	if (!createDescriptorPool()) {
		return false;
	}

	if (!createSceneResources()) {
		return false;
	}

	// При запуске на сцене две пирамиды
	if (!addObject(makeObject(-1.5f, 0.0f, 6.0f, 1.0f, 0.6f, 0.6f)) ||
	    !addObject(makeObject( 1.5f, 0.0f, 6.0f, 0.6f, 0.6f, 1.0f))) {
		return false;
	}

	// Один точечный источник перед пирамидами и один прожектор,
	// который светит сверху-спереди на правую пирамиду
	addPointLight(0.0f, -1.5f, 4.5f, 1.0f, 0.6f, 0.2f, 4.0f);
	addSpotLight(1.5f, -2.0f, 3.0f, 0.0f, 0.6f, 1.0f, 0.4f, 0.6f, 1.0f, 10.0f, 10.0f, 20.0f);

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (Object& object : objects) {
		destroyObjectResources(object);
	}
	objects.clear();

	// Набор сцены уничтожается вместе с пулом
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);

	destroyMappedBuffer(vk_spot_light_buffer, vk_spot_light_buffer_allocation);
	destroyMappedBuffer(vk_point_light_buffer, vk_point_light_buffer_allocation);
	destroyMappedBuffer(vk_scene_uniform_buffer, vk_scene_uniform_buffer_allocation);

	destroyMappedBuffer(vk_index_buffer, vk_index_buffer_allocation);
	destroyMappedBuffer(vk_vertex_buffer, vk_vertex_buffer_allocation);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);
}

void update(double time) {
	// Сколько секунд прошло с прошлого кадра
	const float delta_time = float(time - previous_time);
	previous_time = time;

	for (Object& object : objects) {
		if (object.animation_playing) {
			object.animation_time += delta_time * object.animation_speed;
		}
	}

	// Интерфейс
	// Окно подстраивает высоту под содержимое, но не выше 650 пикселей
	ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, 650));
	ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", camera_position, 0.01f);
		ImGui::SliderFloat("Pitch (deg)", &camera_pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("Yaw (deg)", &camera_yaw, -180.0f, 180.0f);

		if (ImGui::RadioButton("Perspective", use_perspective)) {
			use_perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", !use_perspective)) {
			use_perspective = false;
		}

		if (use_perspective) {
			ImGui::SliderFloat("FOV (deg)", &perspective_fov, 10.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Size", &orthographic_size, 0.5f, 10.0f);
		}
	}

	if (ImGui::CollapsingHeader("Lighting")) {
		// PushID отделяет одинаковые подписи ("Color", "Intensity", ...) в разных разделах
		ImGui::PushID("Ambient");
		ImGui::SeparatorText("Ambient light");
		ImGui::ColorEdit3("Color", ambient_color);
		ImGui::SliderFloat("Intensity", &ambient_intensity, 0.0f, 1.0f);
		ImGui::PopID();

		ImGui::PushID("Directional");
		ImGui::SeparatorText("Directional light");
		ImGui::DragFloat3("Direction", directional_light_direction, 0.01f);
		ImGui::ColorEdit3("Color", directional_light_color);
		ImGui::SliderFloat("Intensity", &directional_light_intensity, 0.0f, 5.0f);
		ImGui::PopID();

		// Удалять источник прямо внутри цикла нельзя (сдвинутся индексы),
		// поэтому запоминаем, какой удалить, и удаляем после цикла
		ImGui::PushID("Point");
		ImGui::SeparatorText("Point lights");
		ImGui::Text("Point lights: %d / %d", int(point_lights.size()), int(max_point_lights));

		ImGui::BeginDisabled(point_lights.size() >= max_point_lights);
		if (ImGui::Button("Add point light")) {
			addPointLight(0.0f, -1.5f, 4.0f, 1.0f, 1.0f, 1.0f, 4.0f);
		}
		ImGui::EndDisabled();

		int remove_point_index = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			ImGui::PushID(point_lights[i].id);
			if (ImGui::TreeNode("Light", "Point light %d", point_lights[i].id)) {
				if (!pointLightInterface(point_lights[i])) {
					remove_point_index = int(i);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_point_index >= 0) {
			point_lights.erase(point_lights.begin() + remove_point_index);
		}
		ImGui::PopID();

		ImGui::PushID("Spot");
		ImGui::SeparatorText("Spot lights");
		ImGui::Text("Spot lights: %d / %d", int(spot_lights.size()), int(max_spot_lights));

		ImGui::BeginDisabled(spot_lights.size() >= max_spot_lights);
		if (ImGui::Button("Add spot light")) {
			addSpotLight(0.0f, -2.0f, 3.0f, 0.0f, 0.6f, 1.0f, 1.0f, 1.0f, 1.0f,
			             15.0f, 10.0f, 20.0f);
		}
		ImGui::EndDisabled();

		int remove_spot_index = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			ImGui::PushID(spot_lights[i].id);
			if (ImGui::TreeNode("Light", "Spot light %d", spot_lights[i].id)) {
				if (!spotLightInterface(spot_lights[i])) {
					remove_spot_index = int(i);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_spot_index >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_spot_index);
		}
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Objects")) {
		ImGui::Text("Objects: %d / %d", int(objects.size()), int(max_objects));

		// Новый объект появляется в центре сцены, ближе к камере
		ImGui::BeginDisabled(objects.size() >= max_objects);
		if (ImGui::Button("Add pyramid")) {
			addObject(makeObject(0.0f, 0.0f, 4.0f, 1.0f, 1.0f, 1.0f));
		}
		ImGui::EndDisabled();

		int remove_index = -1;

		for (size_t i = 0; i < objects.size(); ++i) {
			// У всех объектов одинаковые подписи элементов ("Position", "Albedo", ...),
			// PushID с номером объекта делает их уникальными для ImGui
			ImGui::PushID(objects[i].id);
			if (ImGui::TreeNode("Pyramid", "Pyramid %d", objects[i].id)) {
				if (!objectInterface(objects[i])) {
					remove_index = int(i);
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		if (remove_index >= 0) {
			removeObject(size_t(remove_index));
		}
	}

	ImGui::End();

	// Матрица проекции
	auto& context = graphics::internal::context;

	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	math::Matrix4 projection;
	if (use_perspective) {
		projection = math::Matrix4::perspective(toRadians(perspective_fov), aspect,
		                                        0.1f, 100.0f);
	} else {
		// Видимая область: по высоте [-size, size], по ширине с учётом пропорций окна
		const float h = orthographic_size;
		const float w = orthographic_size * aspect;
		projection = math::Matrix4::orthographic(-w, w, -h, h, 0.1f, 100.0f);
	}

	// Общие данные сцены. Цвет каждого источника умножается на его интенсивность
	SceneUniforms& scene = *vk_scene_uniform_buffer_memory;

	const math::Matrix4 view_projection = projection * viewMatrix();
	memcpy(scene.view_projection, view_projection.elements, sizeof(view_projection));

	for (int i = 0; i < 3; ++i) {
		scene.camera_position[i] = camera_position[i];
		scene.directional_light_direction[i] = directional_light_direction[i];
		scene.directional_light_color[i] = directional_light_color[i] * directional_light_intensity;
		scene.ambient_light[i] = ambient_color[i] * ambient_intensity;
	}

	scene.point_light_count = uint32_t(point_lights.size());
	scene.spot_light_count = uint32_t(spot_lights.size());

	// Массивы источников в storage буферах
	for (size_t i = 0; i < point_lights.size(); ++i) {
		const PointLight& light = point_lights[i];
		PointLightData& data = vk_point_light_buffer_memory[i];

		for (int k = 0; k < 3; ++k) {
			data.position[k] = light.position[k];
			data.color[k] = light.color[k] * light.intensity;
		}
	}

	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLight& light = spot_lights[i];
		SpotLightData& data = vk_spot_light_buffer_memory[i];

		for (int k = 0; k < 3; ++k) {
			data.position[k] = light.position[k];
			data.direction[k] = light.direction[k];
			data.color[k] = light.color[k] * light.intensity;
		}

		// В шейдер передаём косинусы углов, чтобы сравнивать их со скалярным произведением
		data.inner_cutoff = std::cos(toRadians(light.inner_angle));
		data.outer_cutoff = std::cos(toRadians(light.outer_angle));
	}

	// Каждый объект пишет свою матрицу модели и материал в свой uniform буфер
	for (Object& object : objects) {
		ModelUniforms& model = *object.uniform_buffer_memory;

		const math::Matrix4 model_matrix = objectModelMatrix(object);
		memcpy(model.model, model_matrix.elements, sizeof(model_matrix));

		for (int k = 0; k < 3; ++k) {
			model.albedo[k] = object.albedo[k];
			model.specular[k] = object.specular[k];
		}

		model.alpha = object.alpha;
		model.use_vertex_colors = object.use_vertex_colors ? 1.0f : 0.0f;
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.1f, 0.1f, 0.2f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0, .y = 0,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0, .maxDepth = 1,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &vertex_buffer_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	// Набор сцены (set = 0) общий для всех объектов: привязываем один раз
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set,
	                        0, nullptr);

	// Набор модели (set = 1) у каждого объекта свой: привязываем перед каждой отрисовкой
	for (const Object& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        vk_pipeline_layout, 1, 1, &object.descriptor_set,
		                        0, nullptr);

		// Рисуем пирамиду по индексам: 12 индексов = 4 треугольника
		vkCmdDrawIndexed(fd.command_buffer, uint32_t(vertex_count), 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
