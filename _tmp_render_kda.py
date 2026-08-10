from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path("assets/kda")
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(48, True)
F_HEAD = font(30, True)
F_BODY = font(24)
F_SMALL = font(20)
F_FORMULA = font(25)

INK = "#172033"
MUTED = "#5C667A"
BLUE = "#3B6FE2"
TEAL = "#169C8C"
ORANGE = "#E9853D"
PURPLE = "#7B5CC7"
RED = "#D05252"
BG = "#F6F8FC"


def center_text(draw, box, text, fnt, fill=INK, spacing=8):
    x1, y1, x2, y2 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(((x1+x2-w)/2, (y1+y2-h)/2), text, font=fnt, fill=fill, spacing=spacing, align="center")


def rounded(draw, box, fill, outline=None, width=3, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=BLUE, width=7):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    import math
    angle = math.atan2(ey-sy, ex-sx)
    length = 18
    spread = 0.55
    p1 = (ex-length*math.cos(angle-spread), ey-length*math.sin(angle-spread))
    p2 = (ex-length*math.cos(angle+spread), ey-length*math.sin(angle+spread))
    draw.polygon([end, p1, p2], fill=color)


def evolution():
    im = Image.new("RGB", (1900, 850), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 55), "从传统注意力到 KDA：每一步解决什么问题", font=F_TITLE, fill=INK)
    d.text((82, 120), "核心变化：从保存全部历史，转向维护一个可遗忘、可纠错的有限关联记忆", font=F_BODY, fill=MUTED)

    stages = [
        ("Softmax Attention", "保存全部 KV\n逐条比较、精确检索", "问题：缓存与计算\n随序列增长", "#E7EEFF", BLUE),
        ("Linear Attention", "压缩为状态矩阵 S\nS <- S + k v^T", "问题：只会追加\n记忆容易互相干扰", "#E4F6F2", TEAL),
        ("Delta Rule", "先读旧预测\n只写入预测误差", "能力：定向擦除\n覆盖旧的键值关联", "#FFF0E5", ORANGE),
        ("Gated DeltaNet", "整体衰减旧状态\n再执行 Delta 更新", "问题：一个注意力头\n只有一个遗忘速度", "#F0EAFB", PURPLE),
        ("KDA", "逐通道衰减\nD = Diag(α)", "结果：多时间尺度\n细粒度管理有限记忆", "#FFE9E9", RED),
    ]
    x0, y0, w, h, gap = 70, 220, 310, 440, 55
    for i, (name, body, issue, fill, color) in enumerate(stages):
        x = x0 + i*(w+gap)
        rounded(d, (x, y0, x+w, y0+h), fill, color, 4, 26)
        center_text(d, (x+18, y0+22, x+w-18, y0+105), name, F_HEAD, color)
        d.line((x+28, y0+120, x+w-28, y0+120), fill=color, width=3)
        center_text(d, (x+24, y0+140, x+w-24, y0+270), body, F_BODY)
        rounded(d, (x+24, y0+300, x+w-24, y0+410), "#FFFFFF", None, 0, 18)
        center_text(d, (x+35, y0+310, x+w-35, y0+400), issue, F_SMALL, MUTED)
        if i < len(stages)-1:
            arrow(d, (x+w+8, y0+h/2), (x+w+gap-8, y0+h/2), MUTED, 5)

    rounded(d, (250, 720, 1650, 800), "#FFFFFF", "#D8DEEA", 2, 18)
    center_text(d, (270, 728, 1630, 792), "KDA = 通道级遗忘门控 + Delta Rule 纠错写入；它不是 Softmax 权重的近似小改动", F_BODY, INK)
    im.save(OUT / "attention-to-kda.png", quality=95)


def update_flow():
    im = Image.new("RGB", (1700, 1180), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 50), "KDA 的一次状态更新", font=F_TITLE, fill=INK)
    d.text((82, 115), "把复杂公式拆成“遗忘 → 预测 → 纠错 → 读取”四个可理解的动作", font=F_BODY, fill=MUTED)

    # Main vertical flow
    cx = 600
    boxes = [
        (210, "旧状态  S(t-1)", "有限大小的关联记忆矩阵", "#E7EEFF", BLUE),
        (390, "① 按通道遗忘", "S_bar(t) = D(t) S(t-1)，D(t) = Diag(alpha(t))", "#F0EAFB", PURPLE),
        (570, "② 读取旧预测", "v_hat(t) = S_bar(t)^T k(t)", "#E4F6F2", TEAL),
        (750, "③ 计算误差并纠错写入", "S(t) = S_bar(t) + beta(t) k(t) [v(t) - v_hat(t)]^T", "#FFF0E5", ORANGE),
        (930, "④ 使用查询读取", "y(t) = S(t)^T q(t)", "#FFE9E9", RED),
    ]
    bw, bh = 850, 125
    for i, (y, title, subtitle, fill, color) in enumerate(boxes):
        rounded(d, (cx-bw/2, y, cx+bw/2, y+bh), fill, color, 4, 24)
        d.text((cx-bw/2+35, y+21), title, font=F_HEAD, fill=color)
        d.text((cx-bw/2+35, y+72), subtitle, font=F_FORMULA, fill=INK)
        if i < len(boxes)-1:
            arrow(d, (cx, y+bh+7), (cx, boxes[i+1][0]-8), MUTED, 5)

    # Inputs and interpretation callouts
    rounded(d, (1110, 255, 1615, 485), "#FFFFFF", "#D8DEEA", 3, 22)
    d.text((1140, 280), "门控参数", font=F_HEAD, fill=PURPLE)
    d.text((1140, 335), "alpha(t)：每个 key 通道的保留率", font=F_BODY, fill=INK)
    d.text((1140, 385), "beta(t)：当前键值关联的更新强度", font=F_BODY, fill=INK)
    d.text((1140, 435), "两者均由当前输入学习得到", font=F_SMALL, fill=MUTED)
    arrow(d, (1110, 375), (1035, 450), PURPLE, 4)

    rounded(d, (1110, 610, 1615, 870), "#FFFFFF", "#D8DEEA", 3, 22)
    d.text((1140, 635), "Delta Rule 的关键", font=F_HEAD, fill=ORANGE)
    d.text((1140, 695), "不是再次写入完整的 v(t)，", font=F_BODY, fill=INK)
    d.text((1140, 740), "而是写入预测误差：", font=F_BODY, fill=INK)
    d.text((1140, 795), "v(t) - v_hat(t)", font=font(34, True), fill=ORANGE)
    arrow(d, (1110, 760), (1035, 810), ORANGE, 4)

    rounded(d, (1110, 930, 1615, 1055), "#FFFFFF", "#D8DEEA", 3, 22)
    center_text(d, (1130, 945, 1595, 1040), "状态大小不随序列长度增长，\n但历史信息经过了有损压缩", F_BODY, RED)

    d.text((82, 1110), "完整递推：S(t) = [I - beta(t) k(t) k(t)^T] D(t) S(t-1) + beta(t) k(t) v(t)^T", font=F_HEAD, fill=INK)
    im.save(OUT / "kda-update-flow.png", quality=95)


evolution()
update_flow()
