# V2 人工工业零件采集

V2 与 V1/P01 保底场景隔离。V1 配置、构建入口和自动 P01 脚本保持不变。

## 当前已实现门禁

- 独立配置：`simulation/configs/single_bin_scene_v2.json`
- 两台原 Franka：`Arm_A`、`Arm_B`
- 三台原固定相机：`CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP`
- 四个轴件：P01/P02 正立，P03/P04 倒立
- 两颗带真实可见通孔的简化六角螺母：N01/N02
- 两把带平行手柄和开口端的简化扳手：W01/W02
- 2 行 4 列料箱和固定 S11-S24 映射
- 中央提梁与 `BIN_CARRY_TCP`
- 计划满载质量和重心检查
- GT 不进入在线 Observation/Canonical 字段的静态约束

`run_v2_scene_acceptance.py` 当前只执行不依赖 Isaac Sim 的静态契约门禁。
它通过不代表 GUI、物理、IK、抓取或搬运已经通过。

## 必须遵守的阶段顺序

1. 运行 V2 离线契约测试。
2. 使用可见 GUI 构建场景并保存 USD、三路相机图和总览图。
3. 验证两臂显式 HOME、IK、碰撞和交接互锁。
4. 依次练习轴件、倒立轴件纠正、螺母和扳手。
5. 验证空箱、满箱和 20 次满载搬运。
6. 最后才运行正式 Canonical Episode 采集。

在 GUI 验收完成前，不得把静态 PASS 描述为场景正式通过；在完整数据校验完成前，
不得把练习数据标记为可训练数据。
