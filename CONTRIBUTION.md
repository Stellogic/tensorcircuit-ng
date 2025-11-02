# TensorCircuit Lattice 模块开源贡献

> **项目**: TensorCircuit - 面向量子计算的可微分编程框架  
> **贡献者**: @Stellogic  
> **项目周期**: 2025年07月 - 2025年09月  
> **项目来源**: 中科院开源之夏 (OSPP) - 开源软件供应链点亮计划

---

## 📌 快速导航

- **核心代码**: [`tensorcircuit/templates/lattice.py`](./tensorcircuit/templates/lattice.py) 
- **测试代码**: [`tests/test_lattice.py`](./tests/test_lattice.py)
- **性能测试**: [`examples/lattice_neighbor_benchmark.py`](./examples/lattice_neighbor_benchmark.py)

---

## 🎯 项目目标与背景

### 问题
TensorCircuit 缺乏统一的格点几何结构表示,用户需要编写临时脚本定义不同的晶格,导致:
- 代码重复,容易出错
- 缺乏几何信息的显式表示
- 难以可视化和复用

### 解决方案
设计并实现了一个**统一、可扩展、可微分**的 Lattice API,支持:
- 10+ 种常见物理晶格(方格、蜂窝、三角形、Kagome等)
- 跨后端自动微分(JAX/TensorFlow/PyTorch)
- 高性能邻居查找算法
- 完整的可视化支持

---

## 💻 核心技术贡献

### 1. API 设计与实现 (1400+ 行)

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

**代码行数**: 1400+ 行 (核心实现)  
**代码质量**: 
- 完整的 docstring (Google 风格)
- 类型标注覆盖率 100%
- 通过 mypy 静态类型检查

---

### 2. 全面的自动化测试 (2600+ 行, 200+ 用例)

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
- **总测试用例数**: 200+
- **代码覆盖率**: 95%+ (核心 API)
- **参数化测试**: 4 后端 × 50+ 测试 = 200+ 实际执行

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

## 📊 代码贡献统计

| 类型 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **核心实现** | `lattice.py` | 1,400+ | 10+ 晶格类,完整 API |
| **单元测试** | `test_lattice.py` | 2,600+ | 200+ 测试用例 |
| **性能测试** | `lattice_neighbor_benchmark.py` | 150+ | 性能基准测试 |

---

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

