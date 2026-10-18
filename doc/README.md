# ParkingEye 文档入口

> 状态：current ｜ 最后更新：2026-10-18
> 先读顺序：**现状 → 路线 → 指标 → 方法 → 记录**。不要再从编号猜内容。

---

## 一、最先读的三份

1. [`current/status.md`](current/status.md)：项目现在做到哪、下一步是什么、有哪些已知问题。
2. [`current/roadmap.md`](current/roadmap.md)：2026-10 → 2027-04 的半年计划；11 月期中报告的结构与演示脚本。
3. [`current/metrics.md`](current/metrics.md)：**全项目唯一指标总表**；其他文档不得复制关键数字。

---

## 二、权威顺序

发生冲突时，按以下顺序取信：

```
current/status.md      （当前状态）
  > current/roadmap.md （当前计划）
  > current/metrics.md （指标数字）
  > methods/*.md       （方法与操作口径）
  > records/*.md       （带日期的实验/审查快照，只增不改）
  > background/*.md    （背景与入门）
  > archive/           （已过期，仅本地保留，不追踪）
```

`records/` 里的数字如果与 `metrics.md` 冲突，以 `metrics.md` 为准；records 只用于追溯“当时为什么这么判断”。

---

## 三、目录结构

```text
doc/
├── README.md                 # 本文件：入口、权威顺序、旧链接映射
├── current/                  # 只放“现在为真”的内容
│   ├── status.md             # 项目状态与交接
│   ├── roadmap.md            # 半年路线图（含 11 月期中报告）
│   ├── metrics.md            # 唯一指标总表
│   └── assessment.md         # 难度与工程量评估
├── methods/                  # 怎么做，不随状态变化
│   ├── capacity-truth.md     # 容量真值采集与校准
│   ├── poi.md                # POI 空间评估方法
│   ├── annotation.md         # 停车场标注规范
│   └── environment.md        # 开发环境与工具
├── records/                  # 带日期的实验/审查快照
│   ├── imagery-2026-10.md    # 影像质量评估
│   ├── detection-2026-10.md  # DOTA/GE 检测实验
│   ├── vlm-probe-2026-09.md  # DeepSeek 视觉能力探测
│   ├── audit-2026-10-17.md   # 第一轮代码/数据审查
│   └── audit-2026-10-18.md   # 第二轮深度审查
├── background/               # 背景与入门
│   ├── background.md         # 研究背景与立项
│   ├── planning.md           # 数据获取与项目规划
│   └── onboarding.md         # 新人快速入门
└── archive/                  # 本地归档（.gitignore 忽略，不追踪）
    └── eight-week-plan-2026-09.md
```

---

## 四、按场景阅读

| 你要做什么 | 读这些 |
|---|---|
| 第一次接触项目 | `background/onboarding.md` → `current/status.md` → `current/metrics.md` |
| 准备 11 月期中报告 | `current/roadmap.md`（第 2 节）→ `current/status.md` → `current/metrics.md` |
| 跑容量/POI 脚本 | `methods/capacity-truth.md`、`methods/poi.md` → `current/metrics.md` |
| 填容量真值 | `methods/capacity-truth.md`（先复制新文件，绝不覆盖） |
| 查某个指标为什么是这个数 | `current/metrics.md` → `records/` 对应实验 |
| 查历史决策/审查 | `records/audit-2026-10-17.md`、`records/audit-2026-10-18.md` |
| 找已过期计划 | `archive/`（本地；Git 不追踪） |

---

## 五、文档状态规则

每份文档头部尽量包含三行：

```text
> 状态：current | historical | archived
> 最后更新：YYYY-MM-DD
> 权威范围：这份文档能回答什么
```

- **current**：可以据此做决策；数字必须和 `metrics.md` 一致。
- **historical**：带日期的快照，只增不改；结论可能已被推翻，阅读时先看顶部横幅。
- **archived**：过期计划/旧状态；放在 `archive/`，不进 Git。

更新规则：

1. 重跑脚本后，先更新 `current/metrics.md`，再更新 `current/status.md`；
2. 新实验写入 `records/`，不要改旧记录；
3. 过期文档移入 `archive/`，同时在本文的旧链接映射里登记；
4. 不要在多个文档里重复关键数字，只写“见 `current/metrics.md`”。

---

## 六、旧文件名映射（防止旧链接失效）

| 旧路径 | 新路径 |
|---|---|
| `doc/00-研究背景与立项说明.md` | `doc/background/background.md` |
| `doc/01-项目规划与数据获取方案.md` | `doc/background/planning.md` |
| `doc/02-八周执行计划.md` | `doc/current/roadmap.md`（旧文已归档到 `doc/archive/`） |
| `doc/03-开发环境与工具说明.md` | `doc/methods/environment.md` |
| `doc/04-影像质量评估与决策记录.md` | `doc/records/imagery-2026-10.md` |
| `doc/05-DOTA预训练记录.md` | `doc/records/detection-2026-10.md` |
| `doc/06-难度与工程量评估.md` | `doc/current/assessment.md` |
| `doc/07-停车场标注规范-XAnyLabeling.md` | `doc/methods/annotation.md` |
| `doc/08-项目快速入门与技术路线.md` | `doc/background/onboarding.md` |
| `doc/10-DeepSeek视觉模型探测记录.md` | `doc/records/vlm-probe-2026-09.md` |
| `doc/11-代码与数据审查记录.md` | `doc/records/audit-2026-10-17.md` |
| `doc/12-容量真值采集操作说明.md` | `doc/methods/capacity-truth.md` |
| `doc/13-项目状态与交接（2026-10-18）.md` | `doc/current/status.md` |
| `doc/14-POI空间评估说明.md` | `doc/methods/poi.md` |
| `doc/15-代码与路线深度审查（2026-10-18）.md` | `doc/records/audit-2026-10-18.md` |
| `doc/16-半年路线图（2026-10至2027-04）.md` | `doc/current/roadmap.md` |
