# ColorDB 统一存储规范

## 1 概述

ColorDB 是用于存储多材料 3D 打印颜色查找数据的标准格式。每个 ColorDB 文件描述一组颜色通道（palette）、打印参数默认值，以及一个或多个数据分组（section）。每个 section 包含一组颜色条目，每条记录一个 CIELAB 颜色值及其对应的材料堆叠配方。

本规范定义 ColorDB 的逻辑字段结构，**编码形式限定为** JSON（UTF-8）与 MessagePack 两种。加载器 **MUST** 同时支持这两种编码，并按 §10.2 的规则自动检测。详细编码约束见 §10。

未来版本若需引入新的编码（如 CBOR、自研二进制等），**MUST** 通过 `schema_version` bump 引入；本规范**不**预留"其他等价编码"口径。

文件扩展名统一使用 `.colordb`，不区分底层序列化格式。

### 1.1 设计要点与演进预期

**核心设计要点**（每条一行，指向详细章节）：

- `channel_class`（封闭枚举，程序语义分类）+ `display_name`（自由文本，人类展示）双字段设计，分离机器可计算的分类与人类可见的展示（§3 / §12）。
- `normalize` 算法内联定义为 `NFC + Unicode Default Case Folding + White_Space 剥离`，消除不同实现之间的歧义（§4.1）。
- palette 内唯一性键 = `(normalize(display_name), normalize(material))`；`channel_class` 不参与 MUST 级唯一性（§4.2）。
- 跨 ColorDB 合并采用 fail-hard 冲突策略，**不**使用 silent first-wins 或 merge-with-warning（§4.3）。
- 多语言支持（i18n）：可选 `display_name_localized` 字段 + spec 自定义的 locale key 归一化 + 三级 fallback（精确 locale → 语言主标签 → `display_name`）（§5）。
- 消费者语义约束写入规范正文：程序分类 **MUST** 基于 `channel_class`，人类展示 **MUST** 基于 `display_name` / `display_name_localized`（§12）。

**演进预期**（前瞻性措辞，当前版本**不**引入）：

- 未来版本**可能**在 palette 条目中新增独立 `finish` / `effect` 字段，用于区分 Matte / Silk / Wood / Marble 等材质纹理；届时会 bump `schema_version`。
- 未来版本**可能**引入 `material_localized` / `vendor_localized` 等字段，为产品标识 / 厂商名提供多语言支持；届时会 bump `schema_version`。
- 未来版本**可能**定义 `material` canonical form，并将 §4.3 中写入方 `material` 字面量统一的 SHOULD 升级为 MUST；届时会 bump `schema_version`。

### 1.2 术语

本规范中关键词的含义遵循 [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119)：

- **MUST / 必须**：绝对要求。
- **MUST NOT / 禁止**：绝对禁止。
- **SHOULD / 应当**：除非有充分理由，否则应遵守。
- **MAY / 可以**：完全可选。

## 2 文件结构

一个 ColorDB 文件由以下顶层字段组成：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `schema_version` | int | 是 | 规范版本，本版固定为 `1` |
| `name` | string | 是 | 数据库名称 |
| `vendor` | string | 否 | 厂商名称 |
| `material_type` | string | 否 | 材料类型 |
| `palette` | array | 是 | 通道定义（见 §3） |
| `defaults` | object | 是 | 打印参数默认值（见 §6） |
| `meta` | object | 否 | 自由格式元数据（见 §7） |
| `sections` | array | 是 | 数据分组（见 §8） |

完整顶层示例见附录 A。

加载器 **MUST** 忽略本规范未定义的顶层字段。

## 3 Palette

`palette` 是一个有序数组，定义该数据库可用的所有颜色通道。数组中元素的位置决定其索引值（从 0 开始），该索引在配方（recipe）中被引用。

palette 数组的**索引位置**是 `recipe` 元素引用的语义标识。写入方 **MUST NOT** 在不同步更新所有 section 中依赖该索引的 `recipe` 的前提下重排、插入或删除 palette 条目；加载器 **MUST** 按 palette 数组的顺序解释 recipe 值。

### 3.1 通道条目

每个通道条目的字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `channel_class` | string (enum) | 是 | 规范色/外观类别，取值必须是 §3.3 枚举值之一 |
| `display_name` | string | 是 | 默认展示名（fallback base），允许 Unicode（中文 / 厂家命名 / SKU 编号等）。详见 §5 |
| `display_name_localized` | object / map | 否 | `{BCP47_tag: string}`，各语言的本地化展示名。详见 §5 |
| `material` | string | 是 | 材料名称（单字符串，本轮不做 i18n 化，见 §5.5） |
| `hex_color` | string | 否 | 十六进制颜色值，格式 `#RRGGBB`，用于 UI 预览 |

**字段职责**：

- `channel_class` 用于**程序语义分类**（预设筛选、首字母分组、模型 stage 映射、色卡兜底等）。
- `display_name` 与 `display_name_localized` 共同承担**人类展示**（UI label、3MF 材料命名、文件命名、日志）。消费者按 §5.2 的三级 fallback 策略选择展示字符串。
- 详细消费规则见 §12。

### 3.2 约束

- `palette` **MUST** 包含至少 1 个条目，最多 255 个条目。
- 每个条目的 `channel_class` **MUST** 为 §3.3 枚举值之一，大小写敏感；加载器 **MUST** 严格校验。
- 每个条目的 `display_name` **MUST** 为非空字符串。
- 每个条目 **MAY** 提供 `display_name_localized`（`{BCP47_tag: string}` 的 map）承载多语言展示名；具体约束与 fallback 规则见 §5。
- `(normalize(display_name), normalize(material))` 在同一 `palette` 内 **MUST** 唯一（详见 §4.2）。`normalize` 算法定义见 §4.1。
- `channel_class` 与 `display_name_localized` **不** 参与**唯一性 MUST 判定**。
- **SHOULD**：同一 `palette` 内 `channel_class` 不重复。当厂家同时提供两条同色系条目（如两条 `Red`："大红" + "玫红"）时，写入方**应**把其中一条归到更细的类（如 `Pink` / `LightRed` / `Burgundy` / `Maroon` / `DarkRed` 等），而非机械保留两条 `Red`。本规范**允许**重复以便容纳写入方尚未完成细分的异常数据，但**重复 `channel_class` 的 palette 不会命中任何常规预设**（见 §3.5 multiset 精确相等）。
- 索引 255 为保留值，表示 Air（见 §9.2），**MUST NOT** 作为 palette 索引分配。

> ⚠️ **注意**：违反上文 `channel_class` 不重复的 SHOULD，将导致该 palette **无法命中 §3.5 任何命名预设**——这一副作用是 MUST 级的功能丧失。写入方若希望保留预设可匹配性，**SHOULD** 将此条视同 MUST。

### 3.3 规范枚举值

`channel_class` 共 **52 个**合法取值，分 6 层：基础色相、明度变体、命名色、金属色、特殊外观、兜底。所有值采用 **PascalCase**，加载器 **MUST** 严格校验取值（大小写敏感）。

#### 3.3.1 基础色相（13）

覆盖 CMYK / RYBW / RGB 预设筛选所需 + 常见色相：

`Red`、`Orange`、`Yellow`、`Green`、`Cyan`、`Blue`、`Purple`、`Magenta`、`Pink`、`Brown`、`White`、`Gray`、`Black`

#### 3.3.2 明度变体（16）

`Light*` / `Dark*` 前缀表达"比基础色明一档 / 暗一档"：

`LightRed`、`DarkRed`、`LightOrange`、`DarkOrange`、`LightYellow`、`DarkYellow`、`LightGreen`、`DarkGreen`、`LightBlue`、`DarkBlue`、`LightPurple`、`DarkPurple`、`LightBrown`、`DarkBrown`、`LightGray`、`DarkGray`

不包含：

- `LightPink`：浅红语义已由 `Pink` 表达。
- `LightWhite` / `DarkBlack`：无意义。
- `Light/DarkCyan` / `Light/DarkMagenta`：色彩管理原色，一般不做深浅变体；确需则走 §3.3.3 或 `Other`。

#### 3.3.3 命名色（13）

色相学上有独立名称、且在 3D 打印耗材市场常见的扩展色：

`Beige`（米色）、`Ivory`（象牙）、`Cream`（奶油）、`Navy`（藏青）、`Teal`（蓝绿）、`Turquoise`（绿松石）、`Olive`（橄榄绿）、`Khaki`（卡其）、`Mint`（薄荷绿）、`Coral`（珊瑚色）、`Lavender`（薰衣草紫）、`Maroon`（栗红）、`Burgundy`（酒红）

#### 3.3.4 金属色（4）

`Gold`、`Silver`、`Bronze`、`Copper`

#### 3.3.5 特殊外观（5）

耗材具备显著的非色相特征时使用：

`Transparent`（透明）、`Translucent`（半透明）、`Fluorescent`（荧光，含荧光粉 / 荧光绿等）、`Glow`（夜光 / 蓄光）、`Multicolor`（多色 / 渐变 / 双色丝光）

#### 3.3.6 兜底（1）

`Other` —— 以上均无法合理匹配时使用（未分类的木纹、大理石、云石纹等）。

> **finish / texture（Matte / Silk / Wood / Marble 等）不是 `channel_class`**：这些是材质或纹理属性，应写入 `display_name`（如 "Matte Charcoal"）。未来的 spec 版本可能新增独立 `finish` 字段，届时会 bump `schema_version`。

### 3.4 归类决策顺序

选择 `channel_class` 时按下列顺序尝试，**命中即止**：

1. 耗材具有强特殊外观（荧光 / 夜光 / 透明 / 半透明 / 多色渐变）→ §3.3.5。
2. 耗材是金属色（金 / 银 / 青铜 / 铜）→ §3.3.4。
3. 颜色能对应到独立命名色（`Pink` / `Navy` / `Mint` / `Beige` / `Maroon` 等）→ §3.3.1 的 `Pink` 或 §3.3.3。
4. 颜色是 "基础色 + 明一档 / 暗一档" → §3.3.2 的 `Light*` / `Dark*`。
5. 颜色是基础色相 → §3.3.1。
6. 以上均不合适 → `Other`。

### 3.5 标准预设映射

实现 CMYK / RYBW / RGB 等预设筛选 **MUST** 按下表把每个预设展开为 `channel_class` **多重集（multiset）**，并与 palette 的 `channel_class` multiset 做**精确相等**匹配：

| 预设 | 匹配的 `channel_class` multiset |
|------|------------------------------|
| `all`（全部通道） | 任意（不筛选，`all` 不参与 multiset 相等判定） |
| `CMYK` | `{Cyan, Magenta, Yellow, Black}` |
| `CMYW` | `{Cyan, Magenta, Yellow, White}` |
| `CMYKW` | `{Cyan, Magenta, Yellow, Black, White}` |
| `RYBW` | `{Red, Yellow, Blue, White}` |
| `RGBW` | `{Red, Green, Blue, White}` |
| `RYB` | `{Red, Yellow, Blue}` |
| `RGB` | `{Red, Green, Blue}` |

**multiset 相等语义**：

- 把 palette 的 `channel_class` 序列视为 multiset（每个 class 保留出现次数，顺序无关）。
- 与预设定义的 multiset 做精确相等：**palette 条目数 MUST = 预设多重集大小**，且**每个 class 的计数 MUST 一致**。
- 示例：palette = `{Red, Red, Yellow, Blue, White}`（5 条，两条 `Red`） **不** 命中 `RYBW`（4 条通道，`Red` 出现 1 次）。
- 示例：palette = `{Red, Yellow, Blue, White}`（4 条，每 class 一次）命中 `RYBW`。
- 示例：palette = `{Red, Yellow, Blue, White, Cyan}`（5 条，比 `RYBW` 多一条）**不**命中 `RYBW`。

**强约束**：

- 预设匹配 **MUST NOT** 做色系泛化。`LightRed` / `Pink` / `Burgundy` / `Maroon` 即便属于"红色系"，也 **MUST NOT** 命中 `RYBW` 等预设的 `Red`。
- `Other` / §3.3.4 金属色 / §3.3.5 特殊外观从不命中任何命名预设；但 `all` 预设包含它们。
- 重复 `channel_class`（违反 §3.2 的 SHOULD）的 palette **不**会命中任何 `CMYK*` / `RYB*` / `RGB*` 预设；只会命中 `all`。
- "按色系分组"（如"所有红色系"）**不**在本规范范围内，可由消费者实现自行在其 UI 层提供。

### 3.6 `channel_class` 使用规则

- palette 内 **不强制唯一**（MUST 层面），但 **SHOULD** 唯一（见 §3.2 SHOULD 规则）。重复会导致该 palette 无法命中常规预设（§3.5）。
- 当厂家同时提供两条同色系条目时，写入方**应**优先把其中一条归到更细分类（`Pink` / `LightRed` / `Burgundy` / `Maroon` 等），而非并列两条相同 `channel_class`。
- 仅用于**程序分类筛选、预设联动、色卡兜底、模型 stage 映射**等场景。
- **MUST NOT** 用于人类可读展示（一律走 `display_name` / `display_name_localized`，见 §5）。
- 详细消费规则见 §12。

### 3.7 Palette 示例

```json
"palette": [
  {
    "channel_class": "White",
    "display_name": "White",
    "display_name_localized": {"zh-CN": "白色", "ja-JP": "白"},
    "material": "PLA Basic",
    "hex_color": "#ffffff"
  },
  {
    "channel_class": "Red",
    "display_name": "Scarlet Red",
    "display_name_localized": {"zh-CN": "大红", "ja-JP": "スカーレット"},
    "material": "PLA Basic",
    "hex_color": "#e31837"
  },
  {
    "channel_class": "Pink",
    "display_name": "玫红",
    "material": "PLA Basic",
    "hex_color": "#c72164"
  },
  {
    "channel_class": "Fluorescent",
    "display_name": "Fluorescent Pink",
    "display_name_localized": {"zh-CN": "荧光粉"},
    "material": "PLA Basic",
    "hex_color": "#ff45b3"
  },
  {
    "channel_class": "Silver",
    "display_name": "金属银",
    "material": "PLA Matte",
    "hex_color": "#c0c0c0"
  },
  {
    "channel_class": "Other",
    "display_name": "Walnut Wood",
    "display_name_localized": {"zh-CN": "胡桃木纹", "ja-JP": "クルミ木目"},
    "material": "PLA Wood",
    "hex_color": "#6f4e37"
  }
]
```

示例说明：

- 第 1 / 2 / 4 / 6 条：`display_name` 写英文作 fallback，`display_name_localized` 补中文（可选日文）。
- 第 3 / 5 条：厂家只填中文 base，未提供 localized map；合法，英语用户会直接看到中文 base 值（三级 fallback 中的最后一级）。
- 第 3 条 `"玫红"` 归入 `Pink`（而非 `Red`）：遵循 §3.4 决策顺序第 3 步（有独立命名色时优先细类），并满足 §3.2 SHOULD 约束（palette 内 `channel_class` 不重复）。
- 示例严格遵循 §3.4 归类决策顺序：**荧光 → `Fluorescent`**（§3.3.5 特殊外观）、**金属银 → `Silver`**（§3.3.4 金属色）、**木纹无明确色相 → `Other`**（§3.3.6 兜底）。
- 本主示例 palette `channel_class` 序列 = `{White, Red, Pink, Fluorescent, Silver, Other}`，**不**命中任何命名预设（见 §3.5），命中 `all`。

### 3.7.1 合法但不推荐：重复 `channel_class` 示例

本小节演示一个 palette **在 MUST 级上合法、但违反 §3.2 SHOULD** 的场景，用于澄清 multiset 匹配语义下此类 palette 的行为。

```json
"palette": [
  {
    "channel_class": "Red",
    "display_name": "Scarlet Red",
    "display_name_localized": {"zh-CN": "大红"},
    "material": "PLA Basic",
    "hex_color": "#e31837"
  },
  {
    "channel_class": "Red",
    "display_name": "玫红",
    "material": "PLA Basic",
    "hex_color": "#c72164"
  },
  {
    "channel_class": "Yellow",
    "display_name": "Yellow",
    "material": "PLA Basic",
    "hex_color": "#ffd000"
  },
  {
    "channel_class": "Blue",
    "display_name": "Blue",
    "material": "PLA Basic",
    "hex_color": "#1e4cff"
  },
  {
    "channel_class": "White",
    "display_name": "White",
    "material": "PLA Basic",
    "hex_color": "#ffffff"
  }
]
```

行为说明：

- **§3.2 MUST 级唯一性通过**：`(normalize(display_name), normalize(material))` 两两不等（`"Scarlet Red"` vs `"玫红"` 不同；其余条目 display_name 各不相同），加载器**不会**拒绝加载。
- **§3.2 SHOULD 级唯一性违反**：`channel_class` 出现两次 `Red`。写入方**应**把 `"玫红"` 归到更细类（如 `Pink` / `LightRed` / `Burgundy` / `DarkRed` 等）。加载器**可**在此处发出警告，但**不**拒绝加载。
- **§3.5 multiset 预设匹配失败**：该 palette `channel_class` multiset = `{Red×2, Yellow, Blue, White}`，与 `RYBW` 的预设 multiset `{Red, Yellow, Blue, White}` 不相等（`Red` 计数不一致）；**不**命中 `RYBW`，也不命中任何其他 `CMYK*` / `RYB*` / `RGB*` 预设。仅命中 `all`。
- **设计含义**：MUST 唯一性保护文件可加载；SHOULD 唯一性与 multiset 匹配共同保证"推荐写法能命中预设、非推荐写法不会静默命中错误预设"。

## 4 规范化、唯一性与合并

### 4.1 字符串规范化算法（`normalize`）

本规范内联定义 `normalize(s)` 算法，Unicode-aware，支持中文 / 全角字符等。算法分三步，顺序不得调整：

1. **Unicode 规范化**为 NFC（组合字符归一）。参考 [Unicode Normalization Forms (UAX #15)](https://www.unicode.org/reports/tr15/)。
2. **Unicode Default Case Folding**：对全字符串应用 [UCD `CaseFolding.txt`](https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt) 中状态为 `C` 或 `F` 的映射（Default Case Folding）。**不等价**于 ASCII `tolower`，也**不等价**于 locale-sensitive 的 `toLowerCase` / `toLocaleLowerCase`。典型差异：德语 `ß`（U+00DF）casefold → `ss`，而 `"ß".toLowerCase()` 返回自身。
3. **剥离 Unicode 空白字符**：删除所有 Unicode `White_Space` property 为 true 的 codepoint。判定基准：[UCD `PropList.txt`](https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt) 中标记为 `White_Space` 的字符（ASCII 空格 / `\t` / `\n` / `\r` / 全角空格 `U+3000` 等 25 个 codepoint）。本规范 v1 锁定为 **UCD 15.1** 定义的 `White_Space` 码点集合；若未来 UCD 修订此集合，变更 **MUST** 通过提升 `schema_version` 引入，加载器 **MUST NOT** 自行跟踪后续 UCD 版本。
4. 返回剩余字符序列。

**明确范围**：

- `normalize` **不**剥离非 `White_Space` 的 invisible codepoint，包括 `U+200B`（ZERO WIDTH SPACE）、`U+200C` / `U+200D`（ZWNJ / ZWJ）、`U+FEFF`（BOM）、`U+00AD`（SOFT HYPHEN）、各类 bidi marker 等。写入方若在 `display_name` / `material` 中混入此类字符，视为**数据瑕疵**，由调用方在输入端自行清洗；加载器**不**负责修正。
- `normalize` **不**做 NFKC / NFKD / 兼容字符等价折叠（例如半角片假名 `ｱ` 与全角 `ア` 视为**不同**字符）。
- `normalize` **不**做 Unicode confusable / security profile 检测。
- **保留中文、标点、数字**（仅剥空白）：`"大红"` / `"Red"` / `"red"` 之间相互不冲突；`"大-红"` 与 `"大红"` 视为不同条目。
- **不做 ASCII-only 过滤**，避免把中文字符串整体归一成空串，导致唯一性退化。

**参考实现**（示例，**非规范性**，实际实现 **MUST** 保证语义等价于上述 3 步定义）：

| 语言 | NFC | Case Folding | White_Space 剥离 |
|---|---|---|---|
| Python | `unicodedata.normalize("NFC", s)` | `s.casefold()`（**真正** Unicode default casefold） | 遍历 + [`regex`](https://pypi.org/project/regex/) 库的 `\p{White_Space}`（标准 `re` 不支持 Unicode property；`str.isspace()` 覆盖范围**与** `White_Space` **不完全一致**，**不**推荐直接用） |
| C++ / ICU | `icu::Normalizer2::getNFCInstance()` | `u_strFoldCase(..., U_FOLD_CASE_DEFAULT)` | `u_hasBinaryProperty(cp, UCHAR_WHITE_SPACE)` |
| JavaScript | `s.normalize("NFC")` | **原生 `toLowerCase` / `toLocaleLowerCase` 不满足要求**（非 casefold 语义）；需借助 [ICU4X](https://github.com/unicode-org/icu4x) WASM、第三方 `@unicode/case-folding` 类库，或基于 `CaseFolding.txt` 自实现 | `s.replace(/\p{White_Space}/gu, "")`（ES2018+）|
| Rust | `unicode-normalization` crate `.nfc()` | `unicode-case-mapping` crate 或 ICU 绑定（`str::to_lowercase()` **不**是 casefold） | 遍历 + `unicode-properties` crate 的 `is_white_space()` |
| Go | `golang.org/x/text/unicode/norm` `norm.NFC.String(s)` | `golang.org/x/text/cases` 的 `cases.Fold()` | `unicode.IsSpace()`（**注**：Go 的 `unicode.IsSpace` 与 Unicode `White_Space` **范围一致**）|

**跨实现一致性约束**：

- 实现 **MUST** 保证同一输入的多次 `normalize` 结果完全一致。
- 实现 **MUST** 保证 `normalize(a) == normalize(b)` 的判定结果与本节定义的 3 步算法等价；跨语言实现若因标准库差异产生偏差，**MUST** 通过额外处理（自建查表、引入第三方库等）补齐。
- 本规范 **SHOULD** 提供官方一致性测试向量（非本轮范围，后续版本补入附录）。

**Authoring 建议（非规范性）**：写入方 **SHOULD** 在生产侧对 `display_name` / `material` 等人可读字符串做一次更激进的不可见字符剥离，至少包括 `U+200B`–`U+200D`（zero-width）、`U+00AD`（soft hyphen）、`U+FEFF`（BOM），以降低视觉相同但唯一性键不同的条目分裂风险。本规则**不**进入 §11 合法性校验；加载器 **MUST NOT** 因写入方未做此处理而拒载。

### 4.2 palette 内唯一性

- `(normalize(display_name), normalize(material))` 在同一 palette 内 **MUST** 唯一。违反则加载器 **MUST** 拒绝该文件并报告错误（见 V3）。
- `channel_class` **不** 参与 MUST 级唯一性判定（加载时重复不是错误），但 §3.2 的 SHOULD 要求 palette 内 `channel_class` 不重复；违反只发出警告、**不**拒绝加载。
- `display_name_localized` 中的任何内容**不**参与唯一性判定（§5.3）。

### 4.3 跨 ColorDB 合并行为

当实现需要合并多个 ColorDB 的 palette 时：

- 合并键 = `(normalize(display_name), normalize(material))`。
- 若两条输入记录具有**相同合并键**但下列任一字段的值不同，实现 **MUST** 拒绝合并并报错（fail hard），**MUST NOT** 采用 silent first-wins 或 merge-with-warning：
  - `channel_class`
  - `hex_color`
  - `display_name`（原始字符串，**非**归一化）
  - `material`（原始字符串，**非**归一化；典型场景：`"PLA Basic"` vs `"pla basic"` 合并键相同但字面量不同，fail hard）
  - `display_name_localized`（按 §5.1.1 spec-canonical key 比较；见下）
- `display_name_localized` 的合并规则：
  - 两端 map 先各自按 §5.1.1 对 key 做 spec-canonicalize，再做 **locale-wise 并集**。
  - 若同一 spec-canonical locale tag 在两端都存在且值**字面不同**，**MUST** fail hard。
  - 一端提供某 spec-canonical locale、另一端未提供视作"互补"，直接保留该 locale 值。
- 允许同合并键下所有字段（含 `display_name_localized` spec-canonical key 与 value 完全相同）完全一致的多条记录（视作同一条，取一份保留）。
- 调用方 **SHOULD** 在输入端对齐数据（去重、统一字段值）后再调用合并接口。
- 写入方 **SHOULD** 在生产数据端对 `material` 字面量做统一（推荐"标准大小写 + 标准空格"），避免与其他来源合并时被 fail-hard 拦截。具体 canonical form **不**在本规范范围内；未来版本可能将这一 SHOULD 升级为 MUST 并定义 `material` canonical form。

**超出 palette 的合并行为（SHOULD）**：本规范 v1 **不**规定 `sections` / `defaults` / `meta` 等非 palette 字段的合并语义。实现若提供此类合并，**SHOULD** 在其公开文档中明确规则，**SHOULD NOT** silent-merge 或采用不可观察的 "first-wins" 策略，**MUST NOT** 在合并结果中默认丢弃任何 section。未来版本考虑引入完整 merge 语义规范。

## 5 多语言支持（i18n）

ColorDB 面向全球使用场景，palette 条目的人类展示名存在被写入方以单一语言（中文 / 日文 / 德文 / 英文等）填写的情况。本规范通过**两层字段**组合解决多语言展示：`display_name` 作为必填的 fallback base，`display_name_localized` 作为可选的本地化条目 map。

### 5.1 `display_name_localized` 字段结构

- 类型：`object`（JSON）/ `map`（MessagePack）。
- 键：合法 [BCP 47](https://www.rfc-editor.org/rfc/rfc5646) 语言标签，示例：`en`、`en-US`、`zh-CN`、`zh-Hant`、`zh-TW`、`ja-JP`、`de-DE`、`fr-FR`、`ko-KR`。
- 值：非空字符串，允许 Unicode 任意字符。
- 整个字段**可选**；写入方 **MAY** 完全省略，或只提供部分 locale 的条目。
- 本规范**不**限制键集合为 BCP 47 的某个子集；加载器 **MUST** 接受任何符合 BCP 47 语法的 tag。
- `display_name` 与 `display_name_localized` 中某个条目**允许**字符串值重复（如 `display_name = "White"` 且 `display_name_localized["en-US"] = "White"`）；实现不应视为错误。

示例：

```json
{
  "channel_class": "Red",
  "display_name": "Scarlet Red",
  "display_name_localized": {
    "en-US": "Scarlet Red",
    "zh-CN": "大红",
    "zh-TW": "猩紅",
    "ja-JP": "スカーレットレッド",
    "de-DE": "Scharlachrot"
  },
  "material": "PLA Basic",
  "hex_color": "#e31837"
}
```

#### 5.1.1 Locale key 归一化（spec 自定义）

BCP 47 语法允许不同字面量表达**同一**语言标签（例：`en-us` / `EN-US` / `en-US` 均合法且语义相同）。为保证 fallback 查找、merge 冲突判定、同一 map 内唯一性三处语义一致，加载器 **MUST** 在读取 `display_name_localized` 时对每个 key 应用**本规范自定义的 locale key 归一化算法**（下称 "spec-canonicalize"），再参与后续语义判定。

**本算法不是完整的 BCP 47 canonical form**：它**只做** subtag-wise 大小写规整，不做 subtag substitution、不展开 grandfathered / redundant tag、不做 likely subtags 推断、不合并 extlang。与 [RFC 5646 §4.5](https://www.rfc-editor.org/rfc/rfc5646#section-4.5) 定义的 canonical form **不等价**；与 ECMA-402 `Intl.getCanonicalLocales` / ICU `uloc_canonicalize` / Babel `Locale.parse(..).__str__()` 等平台 API **亦不等价**（见下文"平台 API 差异表"）。

**Subtag 大小写规约**：

| Subtag 类型 | 归一化规则 | 示例 |
|---|---|---|
| language（primary） | 全小写 | `en`、`zh`、`ja` |
| extlang | 全小写（3 字母，紧跟 primary language，最多 3 个连续） | `cmn` |
| script | Title Case（首字母大写 + 后三字小写；共 4 字母） | `Hant`、`Latn`、`Cyrl` |
| region | 2 字母字母 → 全大写（ISO 3166-1）；3 字符全数字 → 保留（UN M.49） | `US`、`CN`、`TW`、`419` |
| variant | 全小写 | `rozaj`、`nedis`、`1996` |
| extension（单字母前缀） | 前缀字母全小写；子段全小写 | `u-ca-gregory` |
| privateuse | `x-` 全小写；子段全小写 | `x-foo` |

**规范化算法**（按 subtag **位置**分类，不依赖长度启发式）：

**前置条件**：本算法的输入 **MUST** 已通过 §11 V18 的 BCP 47 语法校验。对语法非法的输入，本算法的行为**未定义**。

1. 按 ASCII `-` 将输入切分为有序 subtag 序列 `[s₁, s₂, ..., sₙ]`。
2. **按位置识别 subtag 类型**：
   - **`s₁` = primary language subtag**：必须是 2–3 字母或 5–8 字母的纯字母 subtag → 全小写。
   - **`s₁` 之后，最多 3 个连续 3-字母纯字母 subtag = extlang**：全小写。
   - **下一段若为 4 字母纯字母 subtag = script**：Title Case。
   - **下一段若为 2 字母纯字母 或 3 字符全数字 subtag = region**：字母 → 全大写；数字 → 保留。
   - **后续每段若为 5–8 字符或"数字开头 4 字符"subtag = variant**：全小写，可重复。
   - **单字母 subtag（除 `x`）启动 extension**：该字母及其后续 `2–8` 字符子段均全小写，直到下一个单字母 subtag 或字符串结束。
   - **`x` 单字母启动 privateuse**：其后所有 `1–8` 字符子段全小写；privateuse **MUST** 位于末尾。
3. 将规整后的 subtag 按原顺序以 `-` 拼接返回。
4. **不**执行以下操作：
   - subtag substitution（`iw` **不**映射为 `he`、`in` **不**映射为 `id`、`ji` **不**映射为 `yi` 等 deprecated 映射）。
   - grandfathered / redundant tag 展开（`i-klingon` **不**映射为 `tlh`、`zh-guoyu` **不**映射为 `cmn`）。
   - extlang 折叠（`zh-cmn-Hans-CN` **不**折叠为 `cmn-Hans-CN`）。
   - likely subtags 推断（`zh-TW` **不**扩展为 `zh-Hant-TW`）。
   - 别名归一（`root` **不**映射为 `und`）。

**同一 map 内去重约束**：若两个字面量 key 经过 spec-canonicalize 后相等（例：`en-US` 与 `EN-US`），整个文件**视为无效**（见 V19）。写入方 **MUST** 在序列化前对 key 做 spec-canonicalize 并去重。

**规范化持久化**：加载器**仅**在内存侧维持 spec-canonical key；**不**要求读取并回写源文件（spec 不规定写入行为，只规定读取语义）。

**写入方建议（SHOULD）**：为降低跨来源合并时的"同语义不同字面量"冲突，写入方 **SHOULD** 使用现代 BCP 47 标签：

- 优先 `he` 而非 `iw`、`id` 而非 `in`、`yi` 而非 `ji`（deprecated 映射）。
- 优先 `zh-Hant` / `zh-Hans` 而非 `zh-TW` / `zh-CN`（当意图指脚本而非地区时）。
- 避免使用 grandfathered 标签（`i-klingon`、`zh-guoyu` 等）。

本规范**不**在 loader 侧强制以上建议；它们属于数据质量指引。

**平台 API 差异表**：下表列出常见平台 locale API 与本规范 spec-canonicalize **多做的事**。实现 **MUST** 裁剪这些副作用（或绕开平台 API 手写 subtag-wise case 归一化）以满足本规范。

| 语言 | 常用 API | 与本规范的差异 | 建议做法 |
|---|---|---|---|
| Python | `babel.Locale.parse(tag).__str__()` | 可能触发 alias 归一、拒绝 grandfathered | 建议**手写**按 §5.1.1 算法分段处理；或使用 `langcodes` 库的 `Language.get(tag).to_tag()` 但显式关掉 substitution |
| Python | `locale.locale_alias` | 含大量 POSIX 兼容别名 | **不**使用 |
| C++ / ICU | `uloc_canonicalize` | 默认做 grandfathered 替换、deprecated 映射 | 改用 `uloc_toLanguageTag(..., strict=true)` 后手动做 subtag case 规整；或完全手写 |
| JavaScript | `Intl.getCanonicalLocales(tag)[0]` | ES2023+ 会做 subtag substitution（`iw`→`he`、`zh-CN`→`zh-Hans-CN` 取决于实现） | **不**直接使用；手写按 `-` 拆分 + case 规整 |
| Rust | `icu_locid::Locale::from_str(tag).to_string()`（`icu_locid` crate） | 解析较纯，但 `icu_locid_transform` 附加组件会做 likely subtags / fallback | **仅**使用 `icu_locid`（不引入 transform 组件），或手写 |
| Go | `golang.org/x/text/language.Parse(tag).String()` | 该库会做 **likely subtags** 推断与 deprecated 替换 | **MUST** 改用字符串级手写实现，或用 `language.Tag.Raw()` 仅取原始 subtag 后再 case 规整 |

**参考伪代码**（非规范性）：

```
function spec_canonicalize(tag):
    parts = tag.split("-")
    result = []
    i = 0
    # primary language
    result.append(parts[i].lower()); i += 1
    # up to 3 extlang
    ext_count = 0
    while i < len(parts) and is_alpha(parts[i]) and len(parts[i]) == 3 and ext_count < 3:
        result.append(parts[i].lower()); i += 1; ext_count += 1
    # optional script (4 letters)
    if i < len(parts) and is_alpha(parts[i]) and len(parts[i]) == 4:
        result.append(title_case(parts[i])); i += 1
    # optional region (2 letters or 3 digits)
    if i < len(parts) and (
        (is_alpha(parts[i]) and len(parts[i]) == 2) or
        (is_digit(parts[i]) and len(parts[i]) == 3)
    ):
        result.append(parts[i].upper() if is_alpha(parts[i]) else parts[i]); i += 1
    # variants, extensions, privateuse: all lowercase, preserve order
    while i < len(parts):
        result.append(parts[i].lower()); i += 1
    return "-".join(result)
```

此伪代码**仅作参考**；实现 **MUST** 保证语义等价于本节定义的 4 步算法。

### 5.2 消费者 fallback 策略

实现在为某用户 locale `L` 渲染一个 palette 条目的展示名时，**MUST** 按以下顺序选取字符串（所有比较**必须**先经过 §5.1.1 spec-canonicalize，再做字面量相等判定）：

1. **Spec-canonicalize 用户 locale**：对 `L` 应用 §5.1.1 算法得到 `spec_canonical(L)`。
2. **精确匹配**：若 `display_name_localized[spec_canonical(L)]` 存在，使用该值。
3. **语言主标签 fallback**（RFC 4647 Lookup 风格）：
   - 从右向左逐段裁剪 `spec_canonical(L)` 的子标签重试（region → script → extlang → language）。
   - 例：用户 `zh-TW` → spec-canonicalize 后仍为 `zh-TW`，若 `zh-TW` 不存在但 `zh` 存在 → 命中 `zh`。
   - 例：用户 `EN-gb` → spec-canonicalize 为 `en-GB`；若 `en-GB` 不存在但 `en` 存在 → 命中 `en`。
   - 注：不同 script 子标签（如 `zh-Hant` vs `zh-CN`）视为不同分支，不做跨 script 匹配；即 `zh-Hant` 的 fallback 链是 `zh-Hant → zh`，**不**经过 `zh-CN`。
4. **回落到 base**：使用 `display_name`。

注意：平台提供的 BCP 47 lookup 实现（如 ICU `uloc_acceptLanguage`、Python `babel.Locale.negotiate`、JavaScript `Intl.Locale` 相关方法）**可能**做 likely subtags 推断或 deprecated 替换，**与本规范 spec-canonicalize 的字面量匹配语义不等价**。实现若使用这些 API，**MUST** 在喂入前将用户 locale 先做 spec-canonicalize，并在库允许的情况下关闭 likely subtags / substitution；无法关闭时 **MUST** 改为手写上述 4 步逻辑。

fallback 过程 **MUST NOT** 修改原始字段内容（只读选择），也 **MUST NOT** 尝试在线翻译或借用其他 palette 条目的内容。

### 5.3 唯一性与规范化

- palette 内唯一性键**只看** `display_name`（配合 `material`），与 §4.2 / V3 一致。
- `display_name_localized` 中的任何内容**不参与**唯一性判定。
- `normalize` 算法（§4.1）**只对** `display_name` 与 `material` 生效；`display_name_localized` 的值**不做** normalize。
- 设计理由：localized 是"装饰字段"，会随时间被补全或增加新 locale；若参与唯一性，补翻译会变成破坏性变更，损害 spec 的演进友好性。

### 5.4 合并行为

（详见 §4.3）跨 ColorDB 合并时：

- 两端的 `display_name_localized` **MUST** 先各自按 §5.1.1 对 key 做 spec-canonicalize，再做 **locale-wise 并集**。
- 若同一 spec-canonical locale tag 在两端都存在且值**字面不同** → **fail hard**（不做 normalize，字面比较；`"大红"` 与 `"大紅"` 视为不同 → fail）。
- 一端有、另一端无该 spec-canonical locale → 保留该值（视为互补翻译）。
- 合并后的 map 仍满足 §5.1.1 的 spec-canonical 唯一性约束。

### 5.5 非 palette 字段的 i18n 立场

本版本**不**做 i18n 化的字段：

- `material`、`vendor`、`material_type`：保持单字符串。这些是产品标识 / 厂商名，跨地区通常以品牌语境使用。
- `meta`：自由格式对象，写入方 **MAY** 自行以 `"description_localized"` 等自定义键承载多语言说明文本，但本规范不规定其格式。

**前瞻性措辞**：未来版本可能引入 `material_localized` / `vendor_localized` 等字段，届时会 bump `schema_version`。

### 5.6 `channel_class` 的 UI 本地化

`channel_class` 的枚举值（`Red` / `Burgundy` / `Fluorescent` 等）是**程序标识符**：

- 在 ColorDB 文件中 **MUST** 保持 PascalCase 英文字面量；加载器大小写敏感校验（§3.3 / V16）。
- 在 UI 展示（色盘筛选标签、分组标题等）时，**消费者**负责翻译到用户 locale；本规范不规定实现方式。
- 本规范**提供**附录 C（非规范性）给出每个 `channel_class` 的 `en-US` / `zh-CN` 推荐翻译作为参考；消费者 **MAY** 直接使用或自建翻译表。
- 若推荐翻译覆盖的 locale 不包含用户所需语言，消费者**应**自行补齐，不应依赖本规范扩展附录 C。

### 5.7 `display_name_localized` 校验

加载器 **MUST** 按 §11 的 V18 与 V19 执行：

- **V18 合法性**：字段存在时，类型必须是 object / map；每个 key 必须是合法 BCP 47 language tag；每个 value 必须是非空字符串。
- **V19 spec-canonical 唯一性**：按 §5.1.1 spec-canonicalize 后，同一 map 内不得出现重复 key。
- 任一条目不合规 → 整个文件视为无效。

### 5.8 fallback 示例

给定 palette 条目：

```json
{"channel_class": "Red", "display_name": "Scarlet Red",
 "display_name_localized": {"en": "Scarlet Red", "zh-CN": "大红"},
 "material": "PLA Basic", "hex_color": "#e31837"}
```

不同用户 locale 下的展示结果：

| 用户 locale | fallback 路径 | 展示结果 |
|---|---|---|
| `zh-CN` | 精确命中 | `"大红"` |
| `zh-TW` | `zh-TW` → `zh`（不存在）→ base | `"Scarlet Red"` |
| `zh-Hant` | `zh-Hant` → `zh`（不存在）→ base | `"Scarlet Red"` |
| `en-US` | `en-US` → `en` 命中 | `"Scarlet Red"` |
| `en-GB` | `en-GB` → `en` 命中 | `"Scarlet Red"` |
| `ja-JP` | `ja-JP` → `ja`（不存在）→ base | `"Scarlet Red"` |
| `de` | `de`（不存在）→ base | `"Scarlet Red"` |

另一条厂家只填中文 base，不提供 `display_name_localized`：

```json
{"channel_class": "Red", "display_name": "玫红", "material": "PLA Basic", "hex_color": "#c72164"}
```

| 用户 locale | 展示结果 |
|---|---|
| 任何 | `"玫红"`（只有 base 可用，跨 locale 一致） |

## 6 Defaults

`defaults` 对象定义打印参数的默认值。section 中未显式设置的参数从此处继承（见 §8.2）。

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `color_layers` | int | 是 | 配方中的颜色层数 |
| `layer_height_mm` | float | 是 | 层高，单位 mm |
| `line_width_mm` | float | 是 | 线宽，单位 mm |
| `base_layers` | int | 是 | 底板层数 |
| `base_channel_idx` | int | 是 | 底板使用的通道索引 |

**类型说明（JSON 编码）**：表中标注为 `float` 的字段，加载器 **MUST** 接受 JSON 整数字面量（如 `0`、`10`）并在内存中统一表示为浮点；标注为 `int` 的字段，加载器 **MUST NOT** 接受浮点字面量，即使数值上是整数（如 `5.0` 对 `color_layers` 非法）。

### 6.1 `base_channel_idx`

**MUST** 为有效的 palette 索引（`0` ≤ 值 < `palette` 长度）。

## 7 Meta

`meta` 是一个可选的自由格式对象，用于存储生成信息、审计记录等非结构化元数据。

- 加载器 **MUST NOT** 依赖 `meta` 中的任何字段进行业务逻辑处理。
- 写入器 **MAY** 在 `meta` 中放置任意键值对。

常见用途包括：`created_at`（生成时间）、`generator`（生成工具名称与版本）、`source_files`（数据来源文件列表）等。

## 8 Sections

`sections` 是一个数组，每个元素是一个数据分组，包含特定打印配置下的颜色条目集合。

### 8.1 Section 字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 数据类型（见 §8.3） |
| `color_layers` | int | 否 | 覆盖 defaults |
| `layer_height_mm` | float | 否 | 覆盖 defaults |
| `line_width_mm` | float | 否 | 覆盖 defaults |
| `base_layers` | int | 否 | 覆盖 defaults |
| `base_channel_idx` | int | 否 | 覆盖 defaults |
| `threshold` | float | 否 | DeltaE 门控阈值（仅 `predicted`） |
| `margin` | float | 否 | 预测需优于实测的余量（仅 `predicted`） |
| `entries` | array | 是 | 颜色条目（见 §9） |

加载器 **MUST** 忽略本规范未定义的 section 级字段。

### 8.2 参数继承

section 中未出现的打印参数字段从 `defaults` 继承。具体规则：

- 对于 `color_layers`、`layer_height_mm`、`line_width_mm`、`base_layers`、`base_channel_idx` 中的每一个：
  - 若 section 中显式设置了该字段，使用 section 的值。
  - 否则，使用 `defaults` 中的值。

section 经继承解析后的参数组合称为该 section 的 **有效配置（resolved config）**。

### 8.3 数据类型

`type` 字段标识该 section 中数据的来源性质：

| 值 | 含义 |
|----|------|
| `"measured"` | 实际测量数据（如从校准板提取） |
| `"predicted"` | 模型预测数据（如通过建模流水线生成） |

`threshold` 和 `margin` 字段 **MUST** 仅在 `type` 为 `"predicted"` 的 section 中出现。`"measured"` 类型的 section **MUST NOT** 包含这两个字段。

### 8.4 唯一性约束

同一文件内，**SHOULD NOT** 存在两个 section 的 **section identity key** 完全相同。section identity key 定义为：

```
(type, color_layers, layer_height_mm, line_width_mm, base_layers)
```

其中每一项取**有效配置值**（即 §8.2 继承 / 覆盖后的 resolved 结果）。

- `base_channel_idx` **不**纳入该 key：它描述通道映射，与打印配置维度正交；两个 section 仅 base 不同时视为同配置的互补数据。
- 规则级别为 **SHOULD**：实现 **MAY** 在检测到重复时发出警告，但 **MUST NOT** 将其视为格式错误而拒绝加载（见 V8）。
- 设计理由：原仅 `(type, color_layers, layer_height_mm)` 三元组过窄；`line_width_mm` 会影响单层出料量、进而影响呈色，`base_layers` 影响底板色。忽略这两项会将合法异构 section 误判为重复。

### 8.5 Sections 示例

一个文件中包含不同层数和类型的多个 section：

```json
"sections": [
  {
    "type": "measured",
    "color_layers": 5,
    "layer_height_mm": 0.08,
    "entries": [ ... ]
  },
  {
    "type": "measured",
    "color_layers": 3,
    "layer_height_mm": 0.12,
    "base_layers": 8,
    "entries": [ ... ]
  },
  {
    "type": "predicted",
    "color_layers": 5,
    "layer_height_mm": 0.08,
    "threshold": 5.0,
    "margin": 0.7,
    "entries": [ ... ]
  }
]
```

## 9 Entry

每条 entry 记录一个颜色值及其对应的材料堆叠配方。

### 9.1 字段

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `lab` | 3 个有限实数（CIELAB `[L, a, b]`） | 是 | 颜色值；具体序列化形态见 §10.3 / §10.4 |
| `recipe` | 无符号 8 位整数序列 | 是 | 材料堆叠配方；具体序列化形态见 §10.3 / §10.4 |

- `lab[0]`（L）：亮度轴；CIELAB 语义上常见范围 `0`–`100`；本规范**不**在 loader 侧强制校验该范围（见 `V20`）。
- `lab[1]`（a）：红绿轴；常见范围约 `-128` ~ `+127`；本规范不强制。
- `lab[2]`（b）：黄蓝轴；常见范围约 `-128` ~ `+127`；本规范不强制。

### 9.2 Recipe

`recipe` 是一个无符号 8 位整数数组，描述每层使用的材料通道。存储顺序固定为自上而下：`recipe[0]` 对应可视面（最上层），`recipe[N-1]` 对应靠近底板的一层。

- `recipe` 的长度 **MUST** 等于所属 section 有效配置中的 `color_layers`。
- 每个元素的有效值：
  - `0` – `254`：palette 通道索引，**MUST** 小于 `palette` 的长度。
  - `255`：保留值，表示 **Air**（该层不打印任何材料）。

### 9.3 去重

同一 section 内，**SHOULD NOT** 出现两条 `recipe` 完全相同的 entry。实现 **MAY** 在检测到重复时发出警告，但 **MUST NOT** 将其视为格式错误而拒绝加载。

## 10 序列化

### 10.1 支持的编码

同一 `.colordb` 扩展名下支持两种序列化编码，逻辑结构完全相同：

| 编码 | 适用场景 |
|------|----------|
| JSON | 人工编辑、调试、小规模数据 |
| MessagePack | 生产部署、大规模数据（更紧凑、解析更快） |

写入器 **MAY** 选择任一编码输出。加载器 **MUST** 同时支持两种编码。

### 10.2 格式检测

加载器 **MUST** 通过文件内容自动判断编码格式，**MUST NOT** 依赖文件扩展名或外部标记区分。

检测规则：读取文件首字节（跳过可选的 UTF-8 BOM `0xEF 0xBB 0xBF`），根据其值判断：

| 首字节 | 编码 |
|--------|------|
| `0x7B`（`{`）或 ASCII 空白（`0x09` `0x0A` `0x0D` `0x20`） | JSON |
| 其他值 | MessagePack |

此规则成立的原因：JSON 顶层对象的首个非空白字符必为 `{`（0x7B），而 MessagePack 的 map 编码首字节为 fixmap（0x80–0x8F）、map16（0xDE）或 map32（0xDF），二者字节范围不重叠。

若首字节既不符合 JSON 开头（`0x7B` 或 ASCII 空白）也不符合 MessagePack map 族开头（`0x80`–`0x8F` / `0xDE` / `0xDF`），加载器 **MUST** 以 "unrecognized-encoding" 错误拒载，**MUST NOT** 尝试解析为第三种格式。

### 10.3 JSON 编码要求

- 文件编码 **MUST** 为 UTF-8。
- **MAY** 包含 UTF-8 BOM。
- 顶层 **MUST** 为 JSON object。
- 数值精度：**写入方 SHOULD** 为 `lab` 中的浮点数保留至少 2 位小数；**加载器 MUST** 无条件接受任意浮点精度（含整数字面量、科学计数法表示、任意小数位数）。

### 10.4 MessagePack 编码要求

- **MUST** 遵循 [MessagePack specification](https://github.com/msgpack/msgpack/blob/master/spec.md)（**2017-09 修订版**或更新版本）定义的编码，特别是 `bin` / `str` 族类型区分。
- 顶层 **MUST** 为 map 类型。
- 字符串 **MUST** 使用 str 族类型（非 bin 族）。
- `entries` 中的 `recipe` 数组 **MAY** 编码为 bin 类型（紧凑字节数组）以提升性能，加载器 **MUST** 同时接受 array 和 bin 两种表示。
- `entries` 中的 `lab` 数组 **MAY** 编码为 bin 类型（3 × float32 小端序），加载器 **MUST** 同时接受 array 和 bin 两种表示。

**Writer 默认选择（SHOULD）**：

- `recipe` **SHOULD** 以 bin 形式写入（节省约 30% 空间）。
- `lab` 若追求紧凑 **MAY** 以 bin 形式写入；若追求精度 **SHOULD** 以 array of float64 写入（bin 形式因固定 float32 会引入精度损失）。

无论写入方如何选择，加载器 **MUST** 兼容两种形式（由 V15 / V25 强制）。

## 11 校验规则

合规的加载器在读取 `.colordb` 文件时 **MUST** 执行以下校验。未通过任何一条 **MUST** 级校验的文件视为无效，加载器 **SHOULD** 拒绝加载并报告错误。

本表只规定**单文件合法性**判定。`channel_class` 可重复性、跨文件合并行为、消费者 fallback 策略等"消费者语义规范"分别落在 §3.6 / §4.3 / §5 / §12 中，不在本表内重复。

**SHOULD 违反的处理**：违反 SHOULD 级规则 **MAY** 由加载器以 `warning` 形式上报，**MUST NOT** 导致加载失败（除非规则条文显式说明其 MUST 级副作用，如 §3.2 的 `channel_class` 唯一性对 §3.5 预设匹配的影响）。

| # | 级别 | 规则 |
|---|------|------|
| V1 | MUST | `schema_version` 等于 `1` |
| V2 | MUST | `palette` 非空且长度 ≤ 255 |
| V3 | MUST | `(normalize(display_name), normalize(material))` 在 palette 内唯一（`normalize` 见 §4.1） |
| V4 | MUST | `defaults` 中所有必需字段存在且类型正确 |
| V5 | MUST | `0 <= defaults.base_channel_idx < palette` 长度 |
| V6 | MUST | 每个 section 包含 `type` 和 `entries` |
| V7 | MUST | `type` 的值为 `"measured"` 或 `"predicted"` |
| V8 | SHOULD | section identity key `(type, color_layers, layer_height_mm, line_width_mm, base_layers)` 在文件内唯一（均取有效配置值，见 §8.4） |
| V9 | MUST | 每条 entry 的 `recipe` 以 **array 形式**出现时，其长度 **MUST** 等于所属 section 的有效 `color_layers`（bin 形式下的长度约束见 V25） |
| V10 | MUST | `recipe` 中每个值 **MUST** 为整数且位于 `[0, 255]` 范围内；且该值 **MUST** 满足 `< palette` 长度 或 等于 `255` |
| V11 | MUST | `threshold` 和 `margin` 仅出现在 `type=predicted` 的 section 中 |
| V12 | SHOULD | 同一 section 内 `recipe` 不重复 |
| V13 | MUST | 加载器忽略未识别的**字段级**项，不视为错误。本规则仅作用于字段级未识别项；文件级合法性（`schema_version` 匹配、`palette` 存在性、`name` / `vendor` / `material_type` 必填等）由 V1 / V3 / V4 独立强制。**MUST NOT** 适用于 `schema_version` 不匹配的情形：若加载器读取到其不支持的 `schema_version`，**MUST** 按 V1 拒载 |
| V14 | MUST | 加载器通过文件内容（§10.2）自动检测编码格式 |
| V15 | MUST | MessagePack 编码时，加载器同时接受 `recipe` 的 array 和 bin 表示 |
| V16 | MUST | palette 中每个条目的 `channel_class` 必填，且为 §3.3 所列枚举值之一（大小写敏感） |
| V17 | MUST | palette 中每个条目的 `display_name` 必填且为非空字符串；该字符串在经 §4.1 `normalize` 处理后 **MUST** 仍至少包含一个非空白、非默认忽略（Default_Ignorable_Code_Point）字符 |
| V18 | MUST | palette 中每个条目的 `display_name_localized`（若存在）**MUST** 为 object / map；每个 key **MUST** 是合法 BCP 47 language tag；每个 value **MUST** 为非空字符串 |
| V19 | MUST | `display_name_localized` 按 §5.1.1 spec-canonicalize 后，同一 map 内不得出现重复 key |
| V20 | MUST | 每条 entry 的 `lab` **MUST** 存在、逻辑上表示 3 个有限实数（`isfinite`；拒绝 `NaN` / `±Infinity`）。JSON 编码时 **MUST** 为长度 3 的 array；MessagePack 编码时 **MUST** 为长度 3 的 array 或长度 12 字节的 bin（见 `V25` / `§10.4`）。JSON 整数字面量 **MUST** 被视为实数；加载器 **MUST** 在内存中统一表示为浮点。本规范**不**强制 Lab 色域范围（L / a / b 的具体取值区间） |
| V21 | MUST | 每个 section 的 `entries` **MUST** 是数组（可为空数组 `[]`，但字段不得缺失；见 V6） |
| V22 | MUST | section 中若提供 override 字段（`color_layers` / `layer_height_mm` / `line_width_mm` / `base_layers` / `base_channel_idx`），类型 **MUST** 与 §6 defaults 字段表一致 |
| V23 | MUST | defaults 和 section override 后的 resolved 配置均满足：`color_layers > 0`、`layer_height_mm > 0`、`line_width_mm > 0`、`base_layers >= 0` |
| V24 | MUST | section override 后的 resolved `base_channel_idx` 仍满足 V5（`0 <= base_channel_idx < palette` 长度） |
| V25 | MUST | MessagePack 编码下：`lab` 若编码为 bin **MUST** 长度 = 12 字节（3 × float32 little-endian）；`recipe` 若编码为 bin **MUST** 长度 = 所属 section 的 resolved `color_layers`（字节数组，每字节代表一个索引） |

## 12 消费者语义约束

本节是规范正文，对所有读取 / 消费 ColorDB 的实现做出约束：

- 凡涉及**程序语义分类**（预设筛选、首字母分组、白色 / 基色定位、模型 stage 映射、色卡兜底等）**MUST** 基于 `channel_class`。**MUST NOT** 依赖 `display_name` 的字面量内容或其任何子串。
- 凡涉及**人类展示**（UI label、3MF 材料命名、文件命名、日志文案）**MUST** 遵循 §5.2 的本地化 fallback 策略：先按 §5.1.1 spec-canonicalize 用户 locale，再按 "精确 spec-canonical locale → 语言主标签 → `display_name`" 三级选取。由于 `display_name` 是必填（V17），fallback 总有值。
- **标准预设筛选** **MUST** 按 §3.5 映射表做 **multiset 精确相等** 匹配；**MUST NOT** 做色系泛化（`LightRed` / `Pink` / `Burgundy` 不命中 `RYBW` 的 `Red`），**MUST NOT** 忽略重复 `channel_class` 的 multiplicity（palette = `{Red×2, Y, B, W}` 不命中 `RYBW`）。
- **跨 ColorDB 合并** **MUST** 遵循 §4.3：合并键 = `(normalize(display_name), normalize(material))`，冲突字段包括 `channel_class` / `hex_color` / `display_name` 原始串 / `material` 原始串 / `display_name_localized`（spec-canonical key 比较，见 §5.1.1），任一不一致 fail hard。
- `channel_class` 到具体字母分组、模型 stage、兜底颜色的**映射表本身**不在本规范范围内；本规范仅规定**源字段**必须是 `channel_class`。
- `channel_class` 的 UI 本地化（色盘筛选标签、分组标题等）由消费者负责翻译；本规范不规定实现方式。消费者 **MAY** 参考附录 C 的推荐翻译，或自建翻译表。

### 12.1 持久化产物与 locale 相关性（实现注意事项）

按 §5.2 选出的展示字符串随用户 locale 变化。若该字符串被用于**持久化产物**（3MF material 名、导出文件名、缓存 key、snapshot / golden file、日志持久化等），产物会具有 locale 相关性。消费者实现须注意：

- **测试与 CI**：**SHOULD** 显式 pin 一个固定 locale（如 `en-US` 或 `C` / `POSIX`），否则跨开发环境 snapshot / golden file 会漂移。
- **文件名用途**：**SHOULD** 对 fallback 结果做安全文件名变换（Unicode → NFKC、替换 / 转义文件系统非法字符如 `/ \ : * ? " < > |` 与控制字符、长度截断等）。本规范**不**规定具体变换规则，由消费者依目标文件系统自定。
- **跨环境交换**：当持久化产物需跨用户 / 机器传递时，消费者 **SHOULD** 记录产物生成时的用户 locale（如写入 `meta` 字段或伴随文件），以便接收方追溯并在必要时重新生成。
- **调用方显式 locale**：消费者 **MAY** 在导出 API 中暴露 `export_locale` 参数以独立于当前 UI locale，但本规范不规定其接口形态。

### 12.2 API 请求字段（非规范性建议）

若实现的 API 协议中提供"按通道筛选"的请求字段（如 `allowed_channels`），**SHOULD** 采用以下任一约定：

- 基于 `channel_class` 的字符串集合；或
- 基于 `(normalize(display_name), normalize(material))` 元组集合。

实现 **SHOULD** 在其接口文档中明确说明所选方案。

## 附录 A 完整示例（非规范性）

```json
{
  "schema_version": 1,
  "name": "CMYW_5L_008",
  "vendor": "BambuLab",
  "material_type": "PLA Basic",

  "palette": [
    {
      "channel_class": "White",
      "display_name": "White",
      "display_name_localized": {"zh-CN": "白色", "ja-JP": "白"},
      "material": "PLA Basic",
      "hex_color": "#ffffff"
    },
    {
      "channel_class": "Cyan",
      "display_name": "Cyan",
      "display_name_localized": {"zh-CN": "青", "ja-JP": "シアン"},
      "material": "PLA Basic",
      "hex_color": "#0086d6"
    },
    {
      "channel_class": "Magenta",
      "display_name": "Magenta",
      "display_name_localized": {"zh-CN": "品红"},
      "material": "PLA Basic",
      "hex_color": "#ec008c"
    },
    {
      "channel_class": "Yellow",
      "display_name": "Yellow",
      "display_name_localized": {"zh-CN": "黄", "de-DE": "Gelb"},
      "material": "PLA Basic",
      "hex_color": "#f4ee2a"
    }
  ],

  "defaults": {
    "color_layers": 5,
    "layer_height_mm": 0.08,
    "line_width_mm": 0.42,
    "base_layers": 10,
    "base_channel_idx": 0
  },

  "meta": {
    "created_at": "2026-04-13T10:30:00Z",
    "generator": "build_colordb v4.0",
    "source_files": ["calibration_page1.png", "calibration_page2.png"]
  },

  "sections": [
    {
      "type": "measured",
      "entries": [
        {"lab": [93.87, -1.51, 1.32],   "recipe": [0, 0, 0, 0, 0]},
        {"lab": [86.15, -10.22, -6.18], "recipe": [0, 0, 0, 0, 1]},
        {"lab": [78.43, -5.67, 12.89],  "recipe": [0, 0, 0, 1, 3]},
        {"lab": [45.21, 52.33, -8.76],  "recipe": [0, 2, 2, 2, 2]}
      ]
    },
    {
      "type": "measured",
      "color_layers": 3,
      "layer_height_mm": 0.12,
      "base_layers": 7,
      "entries": [
        {"lab": [94.12, -1.23, 0.98],   "recipe": [0, 0, 0]},
        {"lab": [82.56, -12.34, -7.89], "recipe": [0, 0, 1]}
      ]
    },
    {
      "type": "predicted",
      "threshold": 5.0,
      "margin": 0.7,
      "entries": [
        {"lab": [72.34, 18.56, -22.11], "recipe": [1, 2, 0, 3, 1]},
        {"lab": [61.89, 35.12, 5.44],   "recipe": [2, 2, 3, 0, 1]},
        {"lab": [55.02, -20.78, 40.33], "recipe": [0, 3, 3, 1, 255]}
      ]
    }
  ]
}
```

上述示例展示了：

- 一个 measured section 继承全部 defaults（5 层、0.08 mm 层高）。
- 一个 measured section 覆盖了 `color_layers`、`layer_height_mm` 和 `base_layers`。
- 一个 predicted section 继承 defaults 并附带 `threshold` / `margin`。
- 最后一条 entry 的 `recipe` 中使用了 `255`（Air）。

**Authoring 建议**：当 `display_name` 作为 fallback 基线时，**SHOULD** 使用 en-US 名称；所有需本地化的语种放入 `display_name_localized`。本示例中 `Magenta` 条目遵循此建议：base 为英文，中文翻译放入 localized map；其他条目（`White` / `Cyan` / `Yellow`）同样以英文为 base 并只提供非英语的 localized 条目。

## 附录 B 默认推荐 hex 色卡（非规范性）

下表为每个 `channel_class` 提供一个典型 hex 色值，供消费者在 palette 条目缺 `hex_color` 时作为兜底显示色使用。

**性质**：

- 本附录为 **informative only**，**MUST NOT** 参与 §11 的合法性校验。
- 写入器 **MAY** 使用这些值作为默认 `hex_color`，也 **MAY** 使用厂家实际测得的值。
- 下述推荐值仅用作"图形化识别"用途，不是精确色度学数据；当具体 hex 与色相学感知存在差异时，以厂家实测值为准。

### B.1 基础色相

| channel_class | 推荐 hex | channel_class | 推荐 hex |
|---|---|---|---|
| `Red` | `#d62828` | `Pink` | `#ffafcc` |
| `Orange` | `#f77f00` | `Brown` | `#6f4e37` |
| `Yellow` | `#fcbf49` | `White` | `#ffffff` |
| `Green` | `#2a9d8f` | `Gray` | `#808080` |
| `Cyan` | `#00b4d8` | `Black` | `#000000` |
| `Blue` | `#1d3557` |  |  |
| `Purple` | `#6a4c93` |  |  |
| `Magenta` | `#d81159` |  |  |

### B.2 明度变体

| channel_class | 推荐 hex | channel_class | 推荐 hex |
|---|---|---|---|
| `LightRed` | `#ff6b6b` | `DarkRed` | `#8b0000` |
| `LightOrange` | `#ffa94d` | `DarkOrange` | `#bf4500` |
| `LightYellow` | `#fff59d` | `DarkYellow` | `#c9a227` |
| `LightGreen` | `#a8e6cf` | `DarkGreen` | `#003f00` |
| `LightBlue` | `#a0d2eb` | `DarkBlue` | `#001f54` |
| `LightPurple` | `#c5a3cf` | `DarkPurple` | `#4a1e5f` |
| `LightBrown` | `#b08968` | `DarkBrown` | `#3d251a` |
| `LightGray` | `#c0c0c0` | `DarkGray` | `#404040` |

### B.3 命名色

| channel_class | 推荐 hex | channel_class | 推荐 hex |
|---|---|---|---|
| `Beige` | `#f5f5dc` | `Mint` | `#b5e4c0` |
| `Ivory` | `#fffff0` | `Coral` | `#ff7f50` |
| `Cream` | `#fffdd0` | `Lavender` | `#b497bd` |
| `Navy` | `#001f3f` | `Maroon` | `#800000` |
| `Teal` | `#008080` | `Burgundy` | `#7b1f3c` |
| `Turquoise` | `#40e0d0` |  |  |
| `Olive` | `#808000` |  |  |
| `Khaki` | `#c3b091` |  |  |

### B.4 金属色

| channel_class | 推荐 hex |
|---|---|
| `Gold` | `#d4af37` |
| `Silver` | `#b8c0c6` |
| `Bronze` | `#cd7f32` |
| `Copper` | `#b87333` |

### B.5 特殊外观

特殊外观的视觉本质无法用单一 hex 准确表达；以下推荐值仅用作占位显示，消费者 **MAY** 使用棋盘纹、渐变、Checker 图案等增强方式替代。

| channel_class | 推荐 hex | 备注 |
|---|---|---|
| `Transparent` | `#f0f0f0` | 建议配合棋盘纹背景渲染 |
| `Translucent` | `#e0e0e0` | 建议配合棋盘纹背景渲染 |
| `Fluorescent` | `#ff00ff` | 典型荧光粉；具体色以厂家 `hex_color` 为准 |
| `Glow` | `#ccffcc` | 典型夜光绿；不反映夜光效果本身 |
| `Multicolor` | `#c0c0c0` | 无法单色表达；建议以渐变或 Checker 图案替代 |

### B.6 兜底

| channel_class | 推荐 hex |
|---|---|
| `Other` | `#808080` |

## 附录 C `channel_class` UI 翻译推荐（非规范性）

为方便消费者本地化 `channel_class` 在 UI 中的显示标签（如色盘筛选按钮、分组标题），本附录提供 52 项枚举值的 `en-US` / `zh-CN` 推荐翻译。

**性质**：

- 本附录为 **informative only**，**MUST NOT** 参与 §11 的合法性校验。
- 消费者 **MAY** 直接使用这些翻译，也 **MAY** 自建翻译表。
- 本表只覆盖 `en-US` 与 `zh-CN` 两种 locale；其他语言（`ja-JP` / `de-DE` / `fr-FR` / `ko-KR` 等）由消费者自行补齐，**不**依赖本规范扩展。
- 写入方在 palette 条目内填 `display_name_localized` 时，**不**必使用本表的翻译（因为 `display_name` 是条目级展示名，非 `channel_class` 级）。

### C.1 基础色相

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `Red` | Red | 红 |
| `Orange` | Orange | 橙 |
| `Yellow` | Yellow | 黄 |
| `Green` | Green | 绿 |
| `Cyan` | Cyan | 青 |
| `Blue` | Blue | 蓝 |
| `Purple` | Purple | 紫 |
| `Magenta` | Magenta | 品红 |
| `Pink` | Pink | 粉 |
| `Brown` | Brown | 棕 |
| `White` | White | 白 |
| `Gray` | Gray | 灰 |
| `Black` | Black | 黑 |

### C.2 明度变体

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `LightRed` | Light Red | 浅红 |
| `DarkRed` | Dark Red | 深红 |
| `LightOrange` | Light Orange | 浅橙 |
| `DarkOrange` | Dark Orange | 深橙 |
| `LightYellow` | Light Yellow | 浅黄 |
| `DarkYellow` | Dark Yellow | 深黄 |
| `LightGreen` | Light Green | 浅绿 |
| `DarkGreen` | Dark Green | 深绿 |
| `LightBlue` | Light Blue | 浅蓝 |
| `DarkBlue` | Dark Blue | 深蓝 |
| `LightPurple` | Light Purple | 浅紫 |
| `DarkPurple` | Dark Purple | 深紫 |
| `LightBrown` | Light Brown | 浅棕 |
| `DarkBrown` | Dark Brown | 深棕 |
| `LightGray` | Light Gray | 浅灰 |
| `DarkGray` | Dark Gray | 深灰 |

### C.3 命名色

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `Beige` | Beige | 米色 |
| `Ivory` | Ivory | 象牙色 |
| `Cream` | Cream | 奶油色 |
| `Navy` | Navy | 藏青 |
| `Teal` | Teal | 蓝绿 |
| `Turquoise` | Turquoise | 绿松石色 |
| `Olive` | Olive | 橄榄绿 |
| `Khaki` | Khaki | 卡其色 |
| `Mint` | Mint | 薄荷绿 |
| `Coral` | Coral | 珊瑚色 |
| `Lavender` | Lavender | 薰衣草紫 |
| `Maroon` | Maroon | 栗红 |
| `Burgundy` | Burgundy | 酒红 |

### C.4 金属色

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `Gold` | Gold | 金 |
| `Silver` | Silver | 银 |
| `Bronze` | Bronze | 青铜 |
| `Copper` | Copper | 铜 |

### C.5 特殊外观

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `Transparent` | Transparent | 透明 |
| `Translucent` | Translucent | 半透明 |
| `Fluorescent` | Fluorescent | 荧光 |
| `Glow` | Glow-in-the-Dark | 夜光 |
| `Multicolor` | Multicolor | 多色 |

### C.6 兜底

| `channel_class` | `en-US` | `zh-CN` |
|---|---|---|
| `Other` | Other | 其他 |
