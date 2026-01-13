# Tensor Library (C++)

**UTEC** – Implementación mínima de tensores **1D/2D/3D** con memoria contigua, operadores, reshape y transformaciones.

---

## 1. Resumen

Esta librería implementa un tensor de hasta **3 dimensiones** usando memoria contigua (`double*`).  
Incluye construcción segura, **Regla de 5** (copia y movimiento), acceso con validación, creadores, operaciones element-wise con **broadcast básico**, cambios de forma **sin copia (move)**, concatenación, producto punto, multiplicación matricial y un sistema de transformaciones por **polimorfismo**.

---

## 2. Características

- Tensores de **1 a 3 dimensiones** (`shape: vector<size_t>`).
- Memoria contigua en heap (`double* data_`) con liberación en destructor.
- **Regla de 5**: constructor de copia, asignación por copia, constructor de movimiento, asignación por movimiento, destructor.
- Acceso con `at(i)`, `at(i,j)`, `at(i,j,k)` y validación de rangos.
- Creadores: `zeros`, `ones`, `random(min,max)`, `arange(start,end)`.
- Operadores: `+`, `-`, `*` (element-wise) y `*` escalar; broadcast básico cuando una dimensión vale `1`.
- `view(new_shape)` y `unsqueeze(dim)` sin copiar datos (mueven el puntero y dejan el original vacío).
- `concat(tensors, dim)`: crea nueva memoria y copia controlada.
- Funciones `friend`: `dot(a,b)` y `matmul(a,b)`.
- Polimorfismo: `TensorTransform` + `apply()` + `ReLU/Sigmoid`.

---

## 3. Estructura de archivos recomendada

```txt
project/
  include/
    Tensor.h
    TensorTransform.h
  src/
    Tensor.cpp
  main.cpp (o tests.cpp)
  CMakeLists.txt (opcional)

