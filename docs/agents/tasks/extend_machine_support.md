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

预期输出：每个机型 × nozzle 一行，格式：

```
Bambu Lab <NEW>           0.2  topology=single  K_process=2  K_filament_raw=2  [OK]  aligned=N fields
Bambu Lab <NEW>           0.4  topology=single  K_process=2  K_filament_raw=2  [OK]  aligned=N fields
```

如果出现 `[MISMATCH]`，说明该机型 K_filament ≠ K_process（A1/A1M/H2D/H2DP/H2C 等）—— 这是已知现象，由脚本 Step 3.5 的 `i%K_filament_raw` 兜底处理。详见 plan v13 §3.1 ⚠ 标记。

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

真机回归至少覆盖 plan v13 §9 步骤 4b 的 5 个 (K_filament, K_process) 组合：(2,2) P2S、(1,2) A1、(2,5) H2D、(4,4) X2D、(2,4) H2C。

## 场景 C：调整 ChromaPrint3D process patch

### C1. 自动重生 draft 草案

```bash
python3 scripts/build_chromaprint_patches.py \
  --bambu-resources BambuStudio/resources/profiles \
  --reference-dir   data/presets \
  --output          /tmp/chromaprint_patches.draft.json
```

> **注意**：原 4 个 P2S 8-slot reference preset 已被删除（plan v13 §4b）。这条命令现在只在保留旧文件的本地分支可运行；future 版本会改为从其他来源获取 reference。

### C2. 手动修改 `chromaprint_patches.json`

直接编辑 `data/presets/chromaprint_patches.json`：

- `process_common`：所有机型 + 所有 nozzle + 所有 face 都应用的 patch。
- `process_per_nozzle.<nozzle>`：仅对该 nozzle 应用。
- `process_per_face.<face>`：仅对该 face 应用，覆盖 per_nozzle / common。
- `filament_common`：当前留空；如果 ChromaPrint3D 未来需要修改 `filament_options_with_variant` 字段，在此处添加。

`$variant_indexed` 字段必须覆盖所有可能出现在 `print_extruder_variant` 中的 variant 名（5 种已知：`Direct Drive Standard` / `Direct Drive High Flow` / `Direct Drive TPU High Flow` / `Bowden Standard` / `Bowden High Flow`）。漏掉某个 variant 会在 `BuildProjectSettings` 运行时 throw `chromaprint_patches.json: $variant_indexed for ... missing variant ...`。

### C3. 跑单测

```bash
python3 -m pytest scripts/tests/test_build_chromaprint_patches.py -v
```

测试会验证：

- 顶层结构完整（`process_common` / `process_per_nozzle` / `process_per_face` / `filament_common`）。
- 所有 `$variant_indexed` 字典覆盖 5 个 variant，没有未解决的 `TODO_*` 占位。
- 字段分类与 BambuStudio `PrintConfig.cpp` 集合一致。

### C4. 真机回归

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

### 问题：H2D × TPU 材料下 `filament_max_volumetric_speed` 不准确

**原因**：plan v13 / LLL —— H2D K_filament=2 < K_process=5，DD TPU HF slot 用 i%2 模式 fallback 到 DD Std 值。这是已知精度损失。

**缓解**：用户在 Bambu Studio 中手动调整即可。ChromaPrint3D 不修改 `filament_options_with_variant` 字段，base 透传。

## 相关文件

- `data/presets/machines.json`：机型注册表（13 机型 × 2 nozzle）
- `data/presets/chromaprint_patches.json`：ChromaPrint3D process patch
- `data/preset_bases/*.json`：26 个离线生成的合并 base preset
- `scripts/build_preset_bases.py`：base 文件生成脚本
- `scripts/build_chromaprint_patches.py`：patch draft 生成脚本
- `scripts/tests/test_build_preset_bases.py`：35 个单测
- `core/src/geo/bambu_preset_catalog.cpp`：catalog 加载实现
- `core/src/geo/bambu_metadata.cpp`：BuildProjectSettings + 字段集合常量
- `core/include/chromaprint3d/bambu_preset_catalog.h`：公开 API
- 完整设计文档：`/home/neroued/.windsurf/plans/multi-machine-preset-redesign-7f831c.md`（v13）
