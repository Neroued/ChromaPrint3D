# 扩展机型支持 / 同步上游 Bambu 资源（维护手册）

本手册覆盖 ChromaPrint3D 多机型预设体系（plan v13）的 3 类常见维护场景：

- **场景 A**：新增一个 BBL 机型（即 `data/presets/machines.json` + `data/preset_bases/*.json`）
- **场景 B**：BambuStudio 上游升级（`PrintConfig.cpp` 字段集合或 system 默认值变化）
- **场景 C**：调整 ChromaPrint3D 的 process patch（修改 `chromaprint_patches.json`）

所有维护操作仅在 **本地** BambuStudio clone 中进行。BambuStudio 源码不进 git（`.gitignore` 已忽略 `/BambuStudio/`），也不进 CI；ChromaPrint3D 仓库只追踪生成产物 `data/preset_bases/*.json`。

## 0. 前置准备

```bash
# 一次性配置：克隆 BambuStudio 到 ChromaPrint3D 仓库根目录
cd /path/to/ChromaPrint3D
git clone --depth 1 https://github.com/bambulab/BambuStudio.git
```

> **注意**：`/BambuStudio/` 已被 `.gitignore` 忽略；不要尝试 `git add BambuStudio`。
> 维护机型需要 BambuStudio 资源（`resources/profiles/BBL.json` + 印机/工艺/耗材 JSON），但仓库本身不依赖 BambuStudio 在 CI 中存在。

## 场景 A：新增机型

### A1. 在 `machines.json` 添加条目

打开 `data/presets/machines.json`，按以下模板添加：

```jsonc
{
  "machines": {
    "Bambu Lab <NEW>": {
      "extruder_topology": "single",   // 或 "dual"
      "nozzles": ["0.2", "0.4"],
      "process_template": {
        "0.2": "0.08mm High Quality @BBL <CODE> 0.2 nozzle",
        "0.4": "0.08mm High Quality @BBL <CODE>"
      },
      "filament_template": {
        "0.2": "Bambu PLA Basic @BBL <CODE> 0.2 nozzle",
        "0.4": "Bambu PLA Basic @BBL <CODE>"
      },
      "printer_template": "Bambu Lab <NEW> {nozzle} nozzle"
    }
  }
}
```

字段说明：

- `extruder_topology`：`"single"` 表示单 extruder（K_process 一般为 2），`"dual"` 表示双 extruder（K_process 通常为 4 或 5）。判断依据：BambuStudio 工艺继承链合并后的 `print_extruder_variant.size()`。
- `process_template[nozzle]`：对应 BambuStudio system 工艺预设名（必须在 `BBL.json` 的 `process_list` 中存在）。
- `filament_template[nozzle]`：对应 BambuStudio system PLA Basic 耗材预设名。允许 alias 到其他机型（如 X1E 0.4 alias 到 `Bambu PLA Basic @BBL X1C`），但脚本会校验该名称必须出现在 `BBL.json` 的 `filament_list` 中。
- `printer_template`：用 `{nozzle}` 占位符，运行时被替换为 `0.2` / `0.4`。

### A2. 验证引用合法性

```bash
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --machines        data/presets/machines.json \
  --validate-only
```

预期输出：每个机型 × nozzle 一行，格式（v13.2）：

```
Bambu Lab <NEW>           0.2  topology=single  K_process=2  K_per_extruder=2  K_filament_raw=2  [OK]  aligned=N fields
Bambu Lab <NEW>           0.4  topology=single  K_process=2  K_per_extruder=2  K_filament_raw=2  [OK]  aligned=N fields
```

K 量说明（v13.2 / m-realfile）：

- **K_process** = `print_extruder_variant.size()`，包含所有 extruder × variant 组合（H2D=5、X2D=4、H2C=4）。
- **K_per_extruder** = `extruder_variant_list[0].split(',').size()`，仅 extruder 0 的 variants。
- **K_filament_raw** = filament inherits 链合并后 `nozzle_temperature.size()`（一般=K_per_extruder或 1）。
- BambuStudio filament 数组长度 = `N × K_per_extruder`（**不是 N×K_process**，源码证据见 plan v13.2 附录 A0a）。

如果出现 `[MISMATCH]`（K_per_extruder ≠ K_filament_raw）：
- X2D：K_filament_raw=4 > K_per_extruder=2，脚本会截断到 K_per_extruder（仅保留 extruder 0 的 variants）—— 这是预期行为（plan v13.2）。
- A1/A1M：K_filament_raw=K_per_extruder=1，一致 ✅。
- H2D/H2DP/H2C：K_filament_raw=K_per_extruder=2，一致 ✅（v13.2 修正后，原 i%2 兜底问题消失）。

如果出现 `error: ... not registered`：检查 `process_template` / `filament_template` / `printer_template` 是否拼写正确。

### A3. 生成 base 文件并提交

```bash
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --machines        data/presets/machines.json \
  --output          data/preset_bases \
  --only            "Bambu Lab <NEW>"
```

脚本会写出 `data/preset_bases/<slug>_0.08mm_n02.json` + `<slug>_0.08mm_n04.json`。

### A4. 跑测试

```bash
# Python 脚本测试
python3 -m pytest scripts/tests/ -v

# C++ 单测（依赖 BambuPresetCatalog 加载新增机型）
CHROMAPRINT3D_DATA_DIR=$(pwd)/data ctest --test-dir build -R SlicerPreset
```

### A5. 真机回归（PR 截图）

参考 plan v13 §9 步骤 4b 的 8 组合表。新增机型至少在 0.4mm 下做 FaceUp + FaceDown 两组手动加载验证（BambuStudio 加载 3MF 不报错、参数与预期一致）。

## 场景 B：BambuStudio 上游升级

### B1. 拉取 BambuStudio 最新代码

```bash
cd BambuStudio && git pull --depth=1 origin master && cd ..
```

### B2. 检查字段集合是否变化

```bash
# 列出新版本中各类字段集合
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --list-print-with-variant-keys > /tmp/print_with_variant_new.txt
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --list-filament-with-variant-keys > /tmp/filament_with_variant_new.txt

# 对比 C++ 端硬编码常量（PrintWithVariantKeys / FilamentWithVariantKeys）
# 文件：core/src/geo/bambu_metadata.cpp
diff /tmp/print_with_variant_new.txt <(grep -oP '"[^"]+"' core/src/geo/bambu_metadata.cpp | grep -A 200 "PrintWithVariantKeys")
```

如果字段集合有变化（新增/删除字段）：

1. 同步 C++ 常量（`core/src/geo/bambu_metadata.cpp` 的 `PrintWithVariantKeys()` / `FilamentWithVariantKeys()`）。
2. 重新生成 26 个 base 文件并 review diff（参见 B3）。

### B3. 重新生成所有 base 文件并 diff

```bash
# 干跑（不写文件，比较是否会产生 diff）
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --machines        data/presets/machines.json \
  --output          data/preset_bases \
  --check

# 真正重写
python3 scripts/build_preset_bases.py \
  --bambu-resources BambuStudio/resources/profiles \
  --machines        data/presets/machines.json \
  --output          data/preset_bases

git diff data/preset_bases/
```

逐文件审查 diff：

- 字段值变化（如 system 默认 `outer_wall_speed` 改了）：通常无害，确认即可。
- 字段新增：检查 ChromaPrint3D 是否需要在 `chromaprint_patches.json` 中显式覆盖。
- 字段删除：上游可能弃用某字段，运行时容错应能跳过。

### B4. 跑测试 + 真机回归

```bash
python3 -m pytest scripts/tests/ -v
CHROMAPRINT3D_DATA_DIR=$(pwd)/data ctest --test-dir build
```

真机回归至少覆盖以下 (K_per_extruder, K_process) 组合（v13.2 / m-realfile）：
- (1, 1) A1/A1M — single extruder + 1 variant
- (2, 2) P2S/X1C/P1S/P1P/X1/X1E/H2S — single extruder + 2 variants
- (2, 4) H2C — dual extruder + 2 variants/extruder
- (2, 5) H2D/H2DP — dual extruder + 不对称（extruder 0 = 2 variants, extruder 1 = 3 variants 含 TPU HF），filament 数组仅用 extruder 0
- (2, 4) X2D — dual extruder × 不同类型（extruder 0 DD, extruder 1 Bowden），filament 数组仅用 extruder 0

## 场景 C：调整 ChromaPrint3D process patch

### C1. 直接编辑 `chromaprint_patches.json`

`data/presets/chromaprint_patches.json` 是 ChromaPrint3D 团队人工维护的产物（plan v13.1 / m3）。原一次性 diff 工具 `scripts/build_chromaprint_patches.py` 已废弃；后续维护直接编辑该 JSON 文件即可。

直接编辑 `data/presets/chromaprint_patches.json`：

- `process_common`：所有机型 + 所有 nozzle + 所有 face 都应用的 patch。
- `process_per_nozzle.<nozzle>`：仅对该 nozzle 应用。
- `process_per_face.<face>`：仅对该 face 应用，覆盖 per_nozzle / common。
- `filament_common`：当前留空；如果 ChromaPrint3D 未来需要修改 `filament_options_with_variant` 字段，在此处添加。

`$variant_indexed` 字段必须覆盖所有可能出现在 `print_extruder_variant` 中的 variant 名（5 种已知：`Direct Drive Standard` / `Direct Drive High Flow` / `Direct Drive TPU High Flow` / `Bowden Standard` / `Bowden High Flow`）。漏掉某个 variant 会在 `BuildProjectSettings` 运行时 throw `chromaprint_patches.json: $variant_indexed for ... missing variant ...`。

### C2. 跑 schema 单测

```bash
python3 -m pytest scripts/tests/test_chromaprint_patches_schema.py -v
```

测试会验证：

- 顶层结构完整（`process_common` / `process_per_nozzle` / `process_per_face` / `filament_common`）。
- 所有 `$variant_indexed` 字典覆盖 5 个 variant，没有未解决的 `TODO_*` 占位。
- `filament_common` 保持为空。

### C3. 真机回归

至少在 P2S × n04 × FaceUp 上验证修改的 patch 在 Bambu Studio 中确实生效（与系统默认值不同）。

## 故障排查

### 问题：BambuStudio 加载 3MF 时 throw `extruder_variant_count != filament_self_index.size()`

**原因**：`filament_extruder_variant` 不等于 `print_extruder_variant × N`（plan v13 / BBB 关键证据）。

**排查**：检查 base 文件中 `filament_extruder_variant` 是否被正确预对齐（应等于 `print_extruder_variant`）。重新跑 `build_preset_bases.py` 应能修复。

### 问题：BambuStudio 切换机型后参数被替换

**原因**：`print_compatible_printers` 没正确生成或没含目标机型。

**排查**：
1. 解压 3MF，查看 `Metadata/project_settings.config` 中的 `print_compatible_printers` 数组。
2. 确认目标机型 `<machine> <nozzle> nozzle` 在数组中。
3. 如果不在：检查 `machines.json` 中两台机型的 `extruder_topology` 是否一致（catalog 仅在同 topology 内联合）。

### 已解决：H2D × TPU 精度问题（v13.2 修正）

**背景**：plan v13 / LLL 原担心 H2D `filament_max_volumetric_speed` 在 DD TPU HF slot 用 i%2 fallback 到 DD Std 值，可能不准。

**v13.2 修正后**：filament 数组长度从 N×K_process 改为 N×K_per_extruder，**TPU HF 仅出现在 `print_extruder_variant`（K_process=5）中，不出现在 filament 数组**。H2D K_per_extruder=2=K_filament_raw，无需 i%2 兜底，问题根本消失。

## 相关文件

- `data/presets/machines.json`：机型注册表（13 机型 × 2 nozzle，含 `slug` 字段作为单一真相源）
- `data/presets/chromaprint_patches.json`：ChromaPrint3D process patch（人工维护）
- `data/preset_bases/*.json`：26 个离线生成的合并 base preset
- `scripts/build_preset_bases.py`：base 文件生成脚本
- `scripts/tests/test_build_preset_bases.py`：base 生成与 schema 单测
- `scripts/tests/test_chromaprint_patches_schema.py`：patch JSON schema 单测
- `core/src/geo/bambu_preset_catalog.cpp`：catalog 加载实现（`LoadFromDir` 一次性扫描 base 缓存 `printer_model`）
- `core/src/geo/bambu_metadata.cpp`：BuildProjectSettings + 字段集合常量
- `core/include/chromaprint3d/bambu_preset_catalog.h`：公开 API
- 完整设计文档：`/home/neroued/.windsurf/plans/multi-machine-preset-redesign-7f831c.md`（v13）+ review-fix 修订 `/home/neroued/.windsurf/plans/multi-machine-preset-review-fixes-411128.md`（v13.1）
