# TensorCircuit Lattice 模块开源贡献

> **项目**: TensorCircuit - 面向量子计算的可微分编程框架  
> **贡献者**: @Stellogic  
> **项目周期**: 2025年07月 - 2025年09月  
> **项目来源**: 中科院开源之夏 (OSPP) - 开源软件供应链点亮计划

---

## 📌 快速导航

### 核心模块
- **Lattice API**: [`tensorcircuit/templates/lattice.py`](./tensorcircuit/templates/lattice.py) - 晶格类与几何工具 (1,524 行)
- **哈密顿量生成**: [`tensorcircuit/templates/hamiltonians.py`](./tensorcircuit/templates/hamiltonians.py) - Heisenberg/Rydberg 哈密顿量 (134 行)
- **单元测试**: [`tests/test_lattice.py`](./tests/test_lattice.py) - 84 测试函数, 14 测试类 (2,236 行)

### 应用示例
- **性能基准**: [`examples/lattice_neighbor_benchmark.py`](./examples/lattice_neighbor_benchmark.py) - KDTree vs 距离矩阵性能对比
- **量子算法**: [`examples/vqe2d_lattice.py`](./examples/vqe2d_lattice.py) - 2D 海森堡模型 VQE 求解
- **几何优化**: [`examples/lennard_jones_optimization.py`](./examples/lennard_jones_optimization.py) - 晶体结构能量最小化

---

## 🎯 项目目标与背景

### 问题
TensorCircuit 缺乏统一的格点几何结构表示,用户需要编写临时脚本定义不同的晶格,导致:
- 代码重复,容易出错
- 缺乏几何信息的显式表示
- 难以可视化和复用

### 解决方案
设计并实现了一个**统一、可扩展、可微分**的 Lattice 生态系统,包含:

**核心几何层**:
- 11 种具体晶格类型(方格、蜂窝、三角形、Kagome等)
- 跨后端自动微分(JAX/TensorFlow/PyTorch)
- 高性能邻居查找算法(支持大规模格点 N>10,000)
- 完整的可视化支持(1D/2D/3D)

**物理应用层**:
- 从晶格几何自动生成稀疏哈密顿量 (Heisenberg, Rydberg 模型)
- 量子门并行化调度算法 (用于 VQE 和 Trotterized 演化)
- 可微分晶格几何优化 (支持能量函数最小化)

---

## 💻 核心技术贡献

### 1. API 设计与实现 (1,524 行)

**文件**: `tensorcircuit/templates/lattice.py`

#### 类层次结构
```python
AbstractLattice (抽象基类)
├── TILattice (周期性晶格)
│   ├── SquareLattice (2D 方格)
│   ├── HoneycombLattice (2D 蜂窝)
│   ├── TriangularLattice (2D 三角)
│   ├── KagomeLattice (2D Kagome)
│   ├── LiebLattice (2D Lieb)
│   ├── CheckerboardLattice (2D 棋盘)
│   ├── RectangularLattice (2D 矩形)
│   ├── ChainLattice (1D 链)
│   ├── DimerizedChainLattice (1D 二聚化链)
│   └── CubicLattice (3D 立方)
└── CustomizeLattice (自定义晶格)
```

#### 核心特性
- ✅ **完全可微分**: 所有几何计算使用 `tc.backend` API,支持自动微分
- ✅ **多后端兼容**: 同一套代码可在 NumPy/JAX/TensorFlow/PyTorch 上运行
- ✅ **周期边界条件**: 正确实现最小镜像约定 (Minimum Image Convention)
- ✅ **动态修改**: `CustomizeLattice` 支持运行时添加/删除格点
- ✅ **高性能优化**: 针对大规模格点(N>200)实现 KDTree 算法优化

#### 关键代码亮点
```python
# 完全向量化的周期边界条件距离计算
def _get_distance_matrix_with_mic_vectorized(self) -> Coordinates:
    """O(N²) 向量化计算,支持 JAX JIT 编译"""
    # 计算所有 3^d 种周期镜像的距离
    # 使用最小镜像约定选择最短距离
    ...

# 智能邻居查找
def _build_neighbors(self, max_k: int = 1, **kwargs):
    """根据格点规模自动选择最优算法"""
    if self.num_sites > 200 and use_kdtree:
        self._build_neighbors_kdtree(max_k, tol)  # O(N log N)
    else:
        self._build_neighbors_by_distance_matrix(max_k, tol)  # O(N²), 可微
```

**代码行数**: 1,524 行 (核心实现)  
**代码质量**: 
- 完整的 docstring (Google 风格)
- 类型标注覆盖率 100%
- 通过 mypy 静态类型检查

---

### 2. 全面的自动化测试 (2,236 行, 84 测试函数)

**文件**: `tests/test_lattice.py`

#### 测试架构

##### 测试类组织
```
TestCustomizeLattice (30+ 测试)
├── 基础功能测试 (初始化、属性访问、迭代器)
├── 邻居查找测试 (NN, NNN, 边界条件)
├── 输入验证测试 (异常处理)
├── 可视化测试 (1D/2D/3D)
└── 动态修改测试 (add_sites, remove_sites)

TestSquareLattice (15+ 测试)
TestHoneycombLattice (10+ 测试)
TestTriangularLattice (8+ 测试)
TestAllTILattices (参数化测试,覆盖 10+ 种晶格)

TestTILattice (边界条件、周期性)
TestLongRangeNeighborFinding (大规模格点, k=7)
TestDistanceMatrix (距离矩阵正确性)
TestBackendIntegration (跨后端测试, 可微分性)
TestApiRobustness (边界情况、异常处理)
TestPrivateHelpers (内部方法单元测试)
```

#### 测试覆盖维度

1. **跨后端测试** (4 个后端 × 50+ 核心测试)
   ```python
   @pytest.mark.parametrize("backend", [
       lf("npb"),    # NumPy
       lf("tfb"),    # TensorFlow
       lf("jaxb"),   # JAX
       lf("torchb")  # PyTorch
   ])
   def test_xxx(self, backend):
       ...
   ```

2. **边界条件测试**
   - 空晶格 (0 个格点)
   - 单格点晶格
   - 断开连接的晶格
   - 重叠坐标
   - 超高维度 (4D+)

3. **数值稳定性测试**
   - 极小距离分离 (ε = 1e-8)
   - 大规模格点 (N = 10000+)
   - 混合边界条件 (PBC + OBC)

4. **可微分性测试** (JAX backend)
   ```python
   def test_tilattice_differentiability(self, jaxb):
       """验证格点参数可微分"""
       def get_total_distance(lattice_constant):
           lat = SquareLattice(size=(5,5), lattice_constant=a)
           return tc.backend.sum(lat.distance_matrix)
       
       grad_val = tc.backend.grad(get_total_distance)(1.5)
       assert grad_val is not None  # 验证梯度计算成功
   ```

5. **性能回归测试**
   - KDTree vs 距离矩阵性能对比
   - 不同规模的邻居查找基准测试

**测试统计**:
- **测试函数数量**: 84+
- **参数化测试**: 4 后端 × ~30 核心测试 ≈ 120+ 实际执行
- **测试类数量**: 14 个测试类,覆盖所有核心功能

---

### 3. 性能优化与工程实践

#### 算法优化

**问题**: 原始 O(N²) 邻居查找算法在大规模格点时性能瓶颈

**解决方案**: 
```python
# 针对不同场景选择最优算法
if self.num_sites > 200 and use_kdtree:
    # 大规模格点: scipy.KDTree O(N log N)
    self._build_neighbors_kdtree(max_k, tol)
else:
    # 小规模/可微分场景: 距离矩阵 O(N²)
    self._build_neighbors_by_distance_matrix(max_k, tol)
```

#### 缓存机制
```python
@property
def distance_matrix(self) -> Coordinates:
    """惰性计算 + 缓存"""
    if self._distance_matrix is None:
        self._distance_matrix = self._compute_distance_matrix()
    return self._distance_matrix
```

#### 工具函数
```python
def get_compatible_layers(bonds: List[Tuple[int, int]]) -> List[List[Tuple]]:
    """贪心边着色算法,用于量子门并行化调度"""
    # 应用场景: Trotterized 哈密顿量演化
    ...
```

---

### 4. 哈密顿量生成模块

**文件**: `tensorcircuit/templates/hamiltonians.py` (134 行)

为了支持晶格上的物理模拟,实现了从几何拓扑自动构建哈密顿量的工具函数:

#### 核心函数

**`heisenberg_hamiltonian(lattice, j_coupling, interaction_scope)`**
- 根据晶格拓扑生成海森堡模型哈密顿量: H = J Σ(X_i X_j + Y_i Y_j + Z_i Z_j)
- 支持各向同性 (单一 J) 和各向异性 (Jx, Jy, Jz) 耦合
- 支持最近邻或全局相互作用
- 输出稀疏矩阵格式,适合大规模量子系统 (L < 20)

**`rydberg_hamiltonian(lattice, omega, delta, c6)`**
- 实现 Rydberg 原子阵列物理哈密顿量
- 包含三项: 驱动 (Ω/2 ΣX_i) + 失谐 (δ Σn_i) + 范德华相互作用 (Σ V_ij n_i n_j)
- 自动从晶格距离矩阵计算长程相互作用 V_ij = C₆/|r_i - r_j|⁶
- 用于模拟中性原子量子计算平台

#### 应用实例: 2D VQE 求解

**文件**: `examples/vqe2d_lattice.py` (99 行)

完整实现了二维方格晶格上的变分量子本征求解器 (VQE):

```python
# 创建 4×4 方格晶格
lattice = SquareLattice(size=(4, 4), pbc=False, precompute_neighbors=1)

# 自动生成海森堡哈密顿量
h = heisenberg_hamiltonian(lattice, j_coupling=[1.0, 1.0, 0.8])

# 获取最近邻键并分组为并行层
nn_bonds = lattice.get_neighbor_pairs(k=1, unique=True)
gate_layers = get_compatible_layers(nn_bonds)

# 使用分层 RZZ/RXX/RYY 门构建 VQE 线路
for layer in gate_layers:
    for j, k in layer:
        c.rzz(j, k, theta=param[idx])
        c.rxx(j, k, theta=param[idx+1])
        c.ryy(j, k, theta=param[idx+2])
```

**技术亮点**:
- 使用 cotengra 优化张量网络收缩
- JAX JIT 编译加速训练循环
- 1000 步优化找到自旋系统基态

---

### 5. 量子线路辅助工具

#### 并行门调度算法

**函数**: `get_compatible_layers(bonds)` (在 `lattice.py` 中)

实现贪心边着色算法,将晶格上的量子门操作分组为可并行层:

**算法思想**:
- 输入: 晶格键列表 [(i, j), ...] (如最近邻键)
- 输出: 分层的键列表,每层内的键互不重叠
- 应用: Trotterized 哈密顿量演化、量子线路深度优化

**示例**:
```python
sq_lattice = SquareLattice(size=(2, 2), pbc=False)
nn_bonds = sq_lattice.get_neighbor_pairs(k=1, unique=True)
gate_layers = get_compatible_layers(nn_bonds)
# 输出: [[(0,1), (2,3)], [(0,2), (1,3)]]
# 第一层: 横向门可并行, 第二层: 纵向门可并行
```

**物理意义**: 在量子硬件上执行时,同一层内的双量子比特门可以同时执行,减少线路深度和退相干影响。

---

### 6. 可微分几何优化

**核心技术**: 全代码使用 `tc.backend` API (而非 NumPy),使晶格几何完全可微分。

**应用场景**: 
- 通过梯度下降优化晶格参数 (如晶格常数、原子位置)
- 最小化物理能量函数 (如 Lennard-Jones 势能)
- 量子设备拓扑优化

#### 实例: Lennard-Jones 晶体结构优化

**文件**: `examples/lennard_jones_optimization.py` (92 行)

使用 JAX 自动微分找到晶格常数的平衡值:

```python
def calculate_potential(log_a):
    """计算晶格的总 Lennard-Jones 势能 (可微分)"""
    lattice_constant = K.exp(log_a)
    
    # 创建晶格 (参数可微)
    lattice = SquareLattice(size=(4,4), lattice_constant=lattice_constant, pbc=True)
    
    # 距离矩阵可微
    d = lattice.distance_matrix
    
    # Lennard-Jones 势能: V = 4ε[(σ/r)¹² - (σ/r)⁶]
    potential_matrix = 4 * epsilon * (
        K.power(sigma / d_safe, 12) - K.power(sigma / d_safe, 6)
    )
    
    return K.sum(potential_matrix) / 2.0

# 使用梯度下降优化
value_and_grad_fun = K.jit(K.value_and_grad(calculate_potential))
optimizer = optax.adam(learning_rate=0.01)

for i in range(200):
    energy, grad = value_and_grad_fun(log_a)
    updates, opt_state = optimizer.update(grad, opt_state)
    log_a = optax.apply_updates(log_a, updates)
```

**物理意义**: 
- 原子间存在短程排斥力 (∝ r⁻¹²) 和长程吸引力 (∝ r⁻⁶)
- 平衡位置对应总势能最小点
- 展示量子计算框架在材料科学中的应用

---

## 📊 代码贡献统计

| 类型 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **核心实现** | `lattice.py` | 1,524 | 13 个晶格类,完整 API |
| **单元测试** | `test_lattice.py` | 2,236 | 84 测试函数, 14 测试类 |
| **哈密顿量模块** | `hamiltonians.py` | 134 | Heisenberg/Rydberg 哈密顿量 |
| **性能测试** | `lattice_neighbor_benchmark.py` | 86 | 邻居查找性能基准 |
| **应用示例** | `vqe2d_lattice.py` | 99 | 2D VQE 完整实现 |
| **优化示例** | `lennard_jones_optimization.py` | 92 | 可微分晶格优化 |

**总计**: 4,171 行 (核心实现 + 测试 + 应用示例)

---

## 🏆 技术栈

**编程语言**: Python 3.8+

**核心依赖**:
- NumPy (数值计算基础)
- JAX (自动微分、JIT 编译)
- TensorFlow (深度学习后端)
- PyTorch (深度学习后端)
- Scipy (KDTree 优化)
- Matplotlib (可视化)

**开发工具**:
- Pytest (测试框架)
- mypy (静态类型检查)
- black (代码格式化)
- Git (版本控制)

---

