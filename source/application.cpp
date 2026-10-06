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
	float color[3];
};

// Треугольная пирамида: вершина сверху (ось y направлена вниз) и основание
// из правильного треугольника в плоскости y = 0.5
// Цвета вершин не задаются вручную, а вычисляются в assignVertexColors()
Vertex vertices[] = {
	{ { +0.000f, -1.0f, +0.0f }, {} }, // 0: вершина пирамиды
	{ { +0.000f, +0.5f, +1.0f }, {} }, // 1: основание, угол 90°
	{ { -0.866f, +0.5f, -0.5f }, {} }, // 2: основание, угол 210°
	{ { +0.866f, +0.5f, -0.5f }, {} }, // 3: основание, угол 330°
};

const size_t vertex_count = sizeof(vertices) / sizeof(vertices[0]);

// 4 грани по 3 индекса, обход по часовой стрелке, если смотреть снаружи
const uint32_t indices[] = {
	0, 2, 1, // боковая грань
	0, 3, 2, // боковая грань
	0, 1, 3, // боковая грань
	1, 2, 3, // основание
};

// Данные одного объекта для шейдеров (зеркалит uniform ObjectUniforms в шейдере)
// По правилам std140 vec4 выравнивается по 16 байтам: matrix занимает байты 0-63,
// color - 64-79, use_vertex_colors - 80-83, поэтому порядок полей совпадает с шейдером
struct ObjectUniforms {
	float matrix[16];
	float color[4];
	float use_vertex_colors;
};

// Один объект на сцене: его параметры из интерфейса и его собственные ресурсы Vulkan
struct Object {
	int id; // постоянный номер объекта для интерфейса

	// Трансформация (доп. задание 2)
	float position[3];
	float rotation[3]; // углы в градусах
	float scale[3];

	// Цвет (доп. задания 4 и 5): умножается на цвета вершин,
	// или используется сам, если они выключены
	float color[4];
	bool use_vertex_colors;

	// Анимация (доп. задание 3): движение по восьмёрке вокруг position и вращение
	bool animation_playing;
	float animation_time;    // "время" анимации, идёт только при воспроизведении
	float animation_speed;   // множитель скорости
	float trajectory_radius; // радиус траектории
	float spin_speed;        // скорость вращения вокруг своей оси, градусов в секунду
	float swing_amplitude;   // амплитуда раскачивания вокруг оси Z, градусы

	// У каждого объекта свой uniform буфер и свой набор дескрипторов,
	// который на этот буфер указывает (доп. задание 6)
	VkBuffer uniform_buffer;
	VmaAllocation uniform_buffer_allocation;
	ObjectUniforms* uniform_buffer_memory;
	VkDescriptorSet descriptor_set;
};

// Сколько объектов максимум можно создать: столько наборов дескрипторов выделяет пул
constexpr uint32_t max_objects = 16;

std::vector<Object> objects;
int next_object_id = 1;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

// Параметры проекции, общие для всей сцены, меняются из интерфейса
bool use_perspective = true;
float perspective_fov = 60.0f;  // вертикальный угол обзора в градусах
float orthographic_size = 4.0f; // половина высоты видимой области

double previous_time = 0.0;

// Объект с настройками по умолчанию в точке (x, y, z) с цветом (r, g, b)
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

	object.color[0] = r;
	object.color[1] = g;
	object.color[2] = b;
	object.color[3] = 1.0f;
	object.use_vertex_colors = true;

	object.animation_playing = false;
	object.animation_time = 0.0f;
	object.animation_speed = 1.0f;
	object.trajectory_radius = 1.0f;
	object.spin_speed = 90.0f;
	object.swing_amplitude = 20.0f;

	return object;
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

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

VkBuffer vk_vertex_buffer;
VmaAllocation vk_vertex_buffer_allocation;
Vertex* vk_vertex_buffer_memory;

VkBuffer vk_index_buffer;
VmaAllocation vk_index_buffer_allocation;
uint32_t* vk_index_buffer_memory;

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

	// Данные извне: один набор дескрипторов с uniform буфером
	const VkPipelineLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
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

bool createBuffers() {
	auto& context = graphics::internal::context;

	// Вершинный буфер
	const VkBufferCreateInfo vertex_buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(vertices),
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo vertex_buffer_allocation = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &vertex_buffer, &vertex_buffer_allocation,
	                    &vk_vertex_buffer, &vk_vertex_buffer_allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate vertex buffer\n";
		return false;
	}

	if (vmaMapMemory(context.allocator, vk_vertex_buffer_allocation,
	                 (void **)&vk_vertex_buffer_memory) != VK_SUCCESS) {
		std::cerr << "Failed to map vertex buffer memory\n";
		return false;
	}

	assignVertexColors();
	memcpy(vk_vertex_buffer_memory, vertices, sizeof(vertices));

	// Индексный буфер
	const VkBufferCreateInfo index_buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(indices),
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo index_buffer_allocation = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &index_buffer, &index_buffer_allocation,
	                    &vk_index_buffer, &vk_index_buffer_allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate index buffer\n";
		return false;
	}

	if (vmaMapMemory(context.allocator, vk_index_buffer_allocation,
	                 (void **)&vk_index_buffer_memory) != VK_SUCCESS) {
		std::cerr << "Failed to map index buffer memory\n";
		return false;
	}

	memcpy(vk_index_buffer_memory, indices, sizeof(indices));

	return true;
}

// Описание набора дескрипторов: в binding = 0 лежит uniform буфер
bool createDescriptorSetLayout() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding descriptor_set_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo descriptor_set_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(descriptor_set_bindings) / sizeof(descriptor_set_bindings[0]),
		.pBindings = descriptor_set_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &descriptor_set_layout, nullptr,
	                                &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	return true;
}

// Пул, из которого выделяются наборы дескрипторов для всех объектов
bool createDescriptorPool() {
	auto& context = graphics::internal::context;

	// Кол-во дескрипторов = N * S (лекция 3, слайд 28): в каждом наборе
	// 1 uniform буфер (N = 1), наборов максимум max_objects (S = max_objects)
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 * max_objects,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		// Разрешаем возвращать в пул наборы удалённых объектов через vkFreeDescriptorSets
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = max_objects,
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

// Создаёт для объекта uniform буфер и набор дескрипторов, который на него указывает
bool createObjectResources(Object& object) {
	auto& context = graphics::internal::context;

	// Uniform буфер объекта
	const VkBufferCreateInfo uniform_buffer = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = (sizeof(ObjectUniforms) + 0xf) & ~0xf, // выравниваем до 16 байт
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo uniform_buffer_allocation = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &uniform_buffer, &uniform_buffer_allocation,
	                    &object.uniform_buffer, &object.uniform_buffer_allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create and allocate uniform buffer\n";
		return false;
	}

	if (vmaMapMemory(context.allocator, object.uniform_buffer_allocation,
	                 (void **)&object.uniform_buffer_memory) != VK_SUCCESS) {
		std::cerr << "Failed to map uniform buffer memory\n";
		return false;
	}

	// Набор дескрипторов объекта, выделяется из общего пула по общему описанию (layout)
	const VkDescriptorSetAllocateInfo descriptor_set = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &descriptor_set,
	                             &object.descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	// Связываем uniform буфер объекта с его набором
	const VkDescriptorBufferInfo uniform_buffer_descriptor = {
		.buffer = object.uniform_buffer,
		.offset = 0,
		.range = sizeof(ObjectUniforms),
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

	vmaUnmapMemory(context.allocator, object.uniform_buffer_allocation);
	vmaDestroyBuffer(context.allocator, object.uniform_buffer, object.uniform_buffer_allocation);
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

// Интерфейс настроек одного объекта
// Возвращает false, если пользователь нажал "Remove"
bool objectInterface(Object& object) {
	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", object.position, 0.01f);
	ImGui::SliderFloat3("Rotation (deg)", object.rotation, -180.0f, 180.0f);
	ImGui::DragFloat3("Scale", object.scale, 0.01f, 0.1f, 5.0f);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Color", object.color);
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

} // namespace

bool initialize() {
	// Layout набора дескрипторов нужен конвейеру, поэтому создаём его первым
	if (!createDescriptorSetLayout()) {
		return false;
	}

	if (!createPipeline()) {
		return false;
	}

	if (!createBuffers()) {
		return false;
	}

	if (!createDescriptorPool()) {
		return false;
	}

	// При запуске на сцене две пирамиды
	if (!addObject(makeObject(-1.5f, 0.0f, 6.0f, 1.0f, 0.6f, 0.6f)) ||
	    !addObject(makeObject( 1.5f, 0.0f, 6.0f, 0.6f, 0.6f, 1.0f))) {
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (Object& object : objects) {
		destroyObjectResources(object);
	}
	objects.clear();

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);

	vmaUnmapMemory(context.allocator, vk_index_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);

	vmaUnmapMemory(context.allocator, vk_vertex_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);
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

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
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

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Objects: %d / %d", int(objects.size()), int(max_objects));

		// Новый объект появляется в центре сцены, ближе к камере
		ImGui::BeginDisabled(objects.size() >= max_objects);
		if (ImGui::Button("Add pyramid")) {
			addObject(makeObject(0.0f, 0.0f, 4.0f, 1.0f, 1.0f, 1.0f));
		}
		ImGui::EndDisabled();

		// Удалять объект прямо внутри цикла нельзя (сдвинутся индексы),
		// поэтому запоминаем, какой удалить, и удаляем после цикла
		int remove_index = -1;

		for (size_t i = 0; i < objects.size(); ++i) {
			// У всех объектов одинаковые подписи элементов ("Position", "Color", ...),
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

	// Матрица проекции, общая для всех объектов
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

	// Каждый объект пишет свою матрицу и свой цвет в свой uniform буфер
	for (Object& object : objects) {
		const math::Matrix4 matrix = projection * objectModelMatrix(object);

		memcpy(object.uniform_buffer_memory->matrix, matrix.elements, sizeof(matrix));
		memcpy(object.uniform_buffer_memory->color, object.color, sizeof(object.color));
		object.uniform_buffer_memory->use_vertex_colors = object.use_vertex_colors ? 1.0f : 0.0f;
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

	// Геометрия у всех объектов общая, а данные (матрица и цвет) у каждого свои:
	// перед каждой отрисовкой привязываем набор дескрипторов этого объекта
	for (const Object& object : objects) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        vk_pipeline_layout, 0, 1, &object.descriptor_set,
		                        0, nullptr);

		// Рисуем пирамиду по индексам: 12 индексов = 4 треугольника
		vkCmdDrawIndexed(fd.command_buffer, sizeof(indices) / sizeof(indices[0]), 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
