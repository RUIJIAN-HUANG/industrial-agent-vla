# ADR-0004：共享图像 CAS 的归属与接口

- 状态：Accepted
- 日期：2026-07-28
- 决策人：A / 项目负责人
- 影响范围：Isaac Adapter、π0.5、OpenVLA-OFT、YOLO、最终 Docker

## 1. 问题

冻结接口用 `cas://sha256/<digest>` 关联同一 RGB 帧，但旧实现只有 URI/SHA
格式校验。真实图像没有统一写入、字节重算、解码和尺寸校验；Mock 模式因此可以
通过测试，真实 VLA 却会收到占位黑图或无法读取图像。

## 2. 决策

采用“公共实现、服务侧调用”：

1. `industrial_agent.image_cas.ImageCas` 是唯一 CAS Writer/Resolver；
2. Isaac Adapter 使用 `write_rgb()` 原子写入，再发布 `ImageReference`；
3. π0.5、OpenVLA-OFT、YOLO 在各自服务入口调用 `resolve_rgb()`；
4. Supervisor 只校验和转发 `ImageReference`，不解码、不转发像素数组；
5. 真实模式解析失败必须 fail-closed，禁止自动创建黑图或 Mock 动作。

三个模型服务不得复制、修改或自行实现另一套路径与 SHA 规则。

## 3. 冻结存储规则

```text
${INDUSTRIAL_AGENT_CAS_ROOT}/
└── sha256/
    └── ab/
        └── abcdef...完整 64 位 digest
```

- 编码：RGB PNG；
- SHA 范围：PNG 编码后的完整文件字节；
- URI：`cas://sha256/<digest>`；
- 摘要：`sha256:<digest>`；
- Producer：读写挂载；
- π0.5、OpenVLA-OFT、YOLO：只读挂载；
- 不允许 HTTP、重定向、`file://`、符号链接或请求方提供的任意路径；
- 文件必须先通过同目录临时文件原子写入，成功后才能发布引用。

## 4. 调用接口

生产端：

```python
reference = image_cas.write_rgb(
    rgb_uint8_hwc,
    camera_id="CAM_A_TOP",
)
```

消费端：

```python
frame = image_cas.resolve_rgb(
    reference,
    expected_camera_id="CAM_A_TOP",
    expected_size=(1280, 720),
)
model_input = frame.rgb
```

返回数组是不可写的 `numpy.uint8[H,W,3]` RGB。缓存只能保存已经完成字节 SHA、
PNG 解码和尺寸检查的帧。

## 5. Docker 挂载

```yaml
services:
  isaac-adapter:
    environment:
      INDUSTRIAL_AGENT_CAS_ROOT: /var/lib/industrial-agent/cas
    volumes:
      - image-cas:/var/lib/industrial-agent/cas

  pi05:
    environment:
      INDUSTRIAL_AGENT_CAS_ROOT: /var/lib/industrial-agent/cas
    volumes:
      - image-cas:/var/lib/industrial-agent/cas:ro

  openvla-oft:
    environment:
      INDUSTRIAL_AGENT_CAS_ROOT: /var/lib/industrial-agent/cas
    volumes:
      - image-cas:/var/lib/industrial-agent/cas:ro

  yolo:
    environment:
      INDUSTRIAL_AGENT_CAS_ROOT: /var/lib/industrial-agent/cas
    volumes:
      - image-cas:/var/lib/industrial-agent/cas:ro

volumes:
  image-cas:
```

Supervisor 不需要挂载 CAS。离线 GT 目录不得挂载到以上在线服务。

## 6. 失败语义

| 错误码 | 情况 | retryable |
|---|---|---:|
| `CAS_1301_NOT_FOUND` | 原子发布后仍找不到文件 | 是 |
| `CAS_1302_DIGEST_MISMATCH` | 字节重算 SHA 不一致 | 否 |
| `CAS_1303_DECODE_FAILED` | 不是安全的 RGB PNG | 否 |
| `CAS_1304_METADATA_MISMATCH` | 相机、宽高或引用不一致 | 否 |
| `CAS_1305_LIMIT_EXCEEDED` | 文件或解码像素超限 | 否 |
| `CAS_1306_UNAVAILABLE` | CAS 根目录或 I/O 不可用 | 是 |

## 7. Definition of Done

- 同一数组重复写入得到相同 digest；
- 写入后恢复的 RGB 与原数组逐像素一致；
- 修改任意一个文件字节后解析失败；
- 声明宽高、相机 ID 不一致时解析失败；
- π0.5、OpenVLA-OFT、YOLO 均使用同一公共 resolver；
- Real 模式不存在 placeholder 或自动 Mock 降级；
- 最终 Docker 中 Producer 为读写卷，三个消费者为只读卷。
