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
```
## 4. Cómo agregar la librería a tu proyecto
4.1 Opción A: Usando CMake (recomendado)

4.1.1Copia los archivos a la estructura mostrada en la sección 3.

4.1.2Usa un CMakeLists.txt como el siguiente:
<code>
cmake_minimum_required(VERSION 3.16)
project(TensorLibDemo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(tensorlib
    src/Tensor.cpp
)
target_include_directories(tensorlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE tensorlib) </code>

3) Compila y ejecuta:
mkdir -p build
cd build
cmake ..
cmake --build .
./demo
4.2 Opción B: Compilación directa (g++)
Si no usas CMake, puedes compilar así (ajusta rutas si cambian):
g++ -std=c++17 -Iinclude main.cpp src/Tensor.cpp -o demo
./demo
5. API de Tensor
5.1 Constructores, asignaciones y destructor
Principio: como se usa new[]/delete[], se implementa la Regla de 5 para evitar fugas o dobles liberaciones.
•	Tensor(): tensor válido vacío (size_=0, data_=nullptr).
•	Tensor(shape, values): valida shape (1..3) y que values.size() == product(shape).
•	Tensor(const Tensor&): copia profunda (deep copy).
•	Tensor& operator=(const Tensor&): copia profunda, maneja self-assignment.
•	Tensor(Tensor&&) noexcept: transfiere ownership del puntero.
•	Tensor& operator=(Tensor&&) noexcept: libera lo actual y toma ownership.
•	~Tensor(): delete[] data_.
5.2 Métodos de consulta
•	shape(): devuelve referencia al vector de dimensiones.
•	dims(): cantidad de dimensiones (1..3).
•	numel(): total de elementos (product(shape)).
5.3 Acceso a elementos
El acceso se realiza por at(...) que utiliza offset(...) y strides_ para calcular la posición lineal en data_. Si hay un índice inválido, se lanza excepción (y opcionalmente se imprime el error si activaste logs).
// 1D
double& at(size_t i);

// 2D (row-major)
double& at(size_t i, size_t j);

// 3D
double& at(size_t i, size_t j, size_t k);
5.4 Creadores estáticos
Devuelven un Tensor por valor (NRVO/move). No usan initializer_list en el código de usuario si no se permite.
static Tensor zeros (const std::vector<size_t>& shape);
static Tensor ones  (const std::vector<size_t>& shape);
static Tensor random(const std::vector<size_t>& shape, double min, double max);
static Tensor arange(long long start, long long end);
5.5 Operadores
Los operadores (+, -, * element-wise) validan compatibilidad de shapes. Se soporta broadcast básico si una dimensión es 1 (por ejemplo (1000x100) + (1x100)).
Tensor operator+(const Tensor& other) const;
Tensor operator-(const Tensor& other) const;
Tensor operator*(const Tensor& other) const; // element-wise
Tensor operator*(double scalar) const;
5.6 Cambios de forma
view(new_shape) y unsqueeze(dim) NO copian datos: transfieren el puntero data_ al tensor resultante. El tensor original queda válido pero vacío (numel() == 0).
Tensor view(const std::vector<size_t>& new_shape);
Tensor unsqueeze(std::size_t dim);
5.7 Concatenación
concat(tensors, dim) crea un tensor nuevo con memoria propia y copia controlada. Todas las dimensiones excepto dim deben coincidir.
static Tensor concat(const std::vector<Tensor>& tensors, std::size_t dim);
5.8 Funciones friend
Se definen como funciones libres con acceso a miembros privados.
friend Tensor dot(const Tensor& a, const Tensor& b);
friend Tensor matmul(const Tensor& a, const Tensor& b);
6. Transformaciones (Polimorfismo)
Para agregar nuevas funciones sin modificar la clase Tensor, se usa una interfaz abstracta TensorTransform con un método virtual apply(x). Tensor::apply(op) recorre el tensor y devuelve uno nuevo con la transformación aplicada elemento a elemento.
class TensorTransform {
public:
  virtual double apply(double x) const = 0;
  virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform { ... };
class Sigmoid : public TensorTransform { ... };

Tensor apply(const TensorTransform& op) const;
7. Ejemplos de uso (sin initializer_list)
7.1 Construcción de shapes con push_back
std::vector<std::size_t> s;
s.push_back(2);
s.push_back(3);

Tensor A = Tensor::ones(s);
7.2 Broadcast con bias
std::vector<std::size_t> xsh; xsh.push_back(2); xsh.push_back(3);
Tensor X = Tensor::ones(xsh);

std::vector<std::size_t> bsh; bsh.push_back(1); bsh.push_back(3);
std::vector<double> bv; bv.push_back(10); bv.push_back(20); bv.push_back(30);
Tensor b(bsh, bv);

Tensor Y = X + b;   // (2x3) + (1x3)
Y.imprimir();
7.3 Matmul
Tensor A = Tensor::ones(shape2(2,3));
Tensor B = Tensor::ones(shape2(3,2));
Tensor C = matmul(A,B);  // 2x2 con 3s
C.imprimir();
7.4 apply(ReLU/Sigmoid)
ReLU relu;
Sigmoid sig;

Tensor A = Tensor::arange(-5, 5);
Tensor B = A.apply(relu);
Tensor C = B.apply(sig);
8. Aplicación final (flujo requerido)
El flujo final típico simula una red simple: X (1000x20x20) -> view(1000x400) -> matmul(W1 400x100) -> + b1 (1x100) -> ReLU -> matmul(W2 100x10) -> + b2 (1x10) -> Sigmoid.
9. Notas de depuración
•	Si random debe ser reproducible, llama srand(seed) al inicio del main.
•	Evita imprimir tensores enormes; imprime solo shapes y algunos valores (por ejemplo la primera fila).
<img width="432" height="634" alt="image" src="https://github.com/user-attachments/assets/e6a219a3-354a-40da-bd73-51fe3590938e" />

