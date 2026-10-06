# Лабораторная работа №1

Основано на коде [vulkan-starter-app](https://github.com/vladeemerr/vulkan-starter-app).

## Сборка
```bash
cmake --preset mingw-debug #windows
cmake --build build-debug --parallel
```
## Запуск

Запускать нужно из корня проекта, иначе не найдутся шейдеры:

```bash
./build-debug/vulkan-starter-app.exe
```
