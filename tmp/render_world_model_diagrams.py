from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parents[1] / "assets" / "world-model"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]
FONT_PATH = next(p for p in FONT_PATHS if p.exists())


def font(size, bold=False):
    bold_path = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    return ImageFont.truetype(str(bold_path if bold and bold_path.exists() else FONT_PATH), size)


NAVY = "#102A43"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#15803D"
ORANGE = "#C2410C"
RED = "#B91C1C"
PURPLE = "#7C3AED"
INK = "#1F2937"
MUTED = "#64748B"
BG = "#F8FAFC"
WHITE = "#FFFFFF"
BORDER = "#CBD5E1"


def rounded(draw, box, fill, outline=BORDER, radius=22, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, box, text, fnt, fill=INK, spacing=8):
    x1, y1, x2, y2 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(((x1 + x2 - w) / 2, (y1 + y2 - h) / 2), text, font=fnt, fill=fill, spacing=spacing, align="center")


def arrow(draw, start, end, color=NAVY, width=5, label=None, label_offset=(0, -35)):
    draw.line([start, end], fill=color, width=width)
    import math
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 16
    p1 = (end[0] - size * math.cos(angle - 0.55), end[1] - size * math.sin(angle - 0.55))
    p2 = (end[0] - size * math.cos(angle + 0.55), end[1] - size * math.sin(angle + 0.55))
    draw.polygon([end, p1, p2], fill=color)
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        draw.text((mx, my), label, font=font(25), fill=color, anchor="mm")


def title(draw, text, subtitle, width):
    draw.text((width / 2, 54), text, font=font(46, True), fill=NAVY, anchor="mm")
    draw.text((width / 2, 105), subtitle, font=font(25), fill=MUTED, anchor="mm")


def architecture():
    w, h = 1800, 1120
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    title(d, "世界模型系统：预测与决策的职责边界", "动作是世界模型的输入；世界模型预测后果，决策模块选择动作", w)

    boxes = {
        "obs": (70, 230, 330, 390),
        "enc": (420, 230, 710, 390),
        "latent": (800, 230, 1070, 390),
        "actor": (1180, 170, 1480, 310),
        "action": (1535, 170, 1740, 310),
        "dyn": (1180, 420, 1480, 600),
        "next": (1535, 430, 1740, 590),
        "heads": (1160, 700, 1510, 900),
        "outputs": (1550, 700, 1740, 900),
        "real": (70, 720, 430, 900),
        "loss": (530, 720, 930, 900),
    }
    rounded(d, boxes["obs"], "#DBEAFE", BLUE); centered(d, boxes["obs"], "真实观察\no_t", font(34, True))
    rounded(d, boxes["enc"], "#E0F2FE", CYAN); centered(d, boxes["enc"], "编码器 / 状态估计\nEφ", font(31, True))
    rounded(d, boxes["latent"], "#EDE9FE", PURPLE); centered(d, boxes["latent"], "潜在状态\nz_t", font(34, True))
    rounded(d, boxes["actor"], "#DCFCE7", GREEN); centered(d, boxes["actor"], "Actor / Planner\n提出或搜索动作", font(29, True))
    rounded(d, boxes["action"], "#DCFCE7", GREEN); centered(d, boxes["action"], "动作\na_t", font(32, True))
    rounded(d, boxes["dyn"], "#FFEDD5", ORANGE); centered(d, boxes["dyn"], "Dynamics\nFθ(z_t, a_t)", font(31, True))
    rounded(d, boxes["next"], "#FFEDD5", ORANGE); centered(d, boxes["next"], "预测状态\nz(t+1)", font(31, True))
    rounded(d, boxes["heads"], "#FEE2E2", RED); centered(d, boxes["heads"], "环境预测头\n奖励 / 终止 / 观察", font(29, True))
    rounded(d, boxes["outputs"], "#FEE2E2", RED); centered(d, boxes["outputs"], "预测奖励\n终止 / 观察", font(27, True))
    rounded(d, boxes["real"], WHITE, NAVY); centered(d, boxes["real"], "真实环境反馈\no_t+1, r_t, d_t", font(31, True))
    rounded(d, boxes["loss"], WHITE, NAVY); centered(d, boxes["loss"], "世界模型损失\n预测值 vs. 真实值", font(31, True))

    arrow(d, (330, 310), (420, 310))
    arrow(d, (710, 310), (800, 310))
    arrow(d, (1070, 280), (1180, 240), color=GREEN)
    arrow(d, (1480, 240), (1535, 240), color=GREEN)
    arrow(d, (1635, 310), (1480, 470), color=ORANGE, label="条件输入", label_offset=(35, -5))
    arrow(d, (1070, 350), (1180, 510), color=ORANGE)
    arrow(d, (1480, 510), (1535, 510), color=ORANGE)
    arrow(d, (1635, 590), (1410, 700), color=RED)
    arrow(d, (1510, 800), (1550, 800), color=RED)
    arrow(d, (250, 720), (250, 390), color=NAVY, label="下一时间步", label_offset=(-75, 0))
    arrow(d, (430, 810), (530, 810), color=NAVY)
    arrow(d, (1550, 860), (930, 860), color=NAVY)
    d.text((900, 1020), "绿色：决策模块    橙/红色：广义世界模型    蓝/紫色：状态表示", font=font(26), fill=MUTED, anchor="mm")
    im.save(OUT / "architecture.png", quality=95)


def training_decision():
    w, h = 1900, 1240
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    title(d, "训练流程与决策流程必须分开理解", "左侧学习“世界规律”，右侧利用规律“选择动作”", w)
    d.line((950, 155, 950, 1160), fill=BORDER, width=4)
    d.text((475, 180), "A. 世界模型训练", font=font(38, True), fill=BLUE, anchor="mm")
    d.text((1425, 180), "B. 想象、规划与执行", font=font(38, True), fill=GREEN, anchor="mm")

    left = [
        ((130, 245, 820, 365), "真实轨迹数据\n(o_t, a_t, r_t, d_t, o_t+1)", "#DBEAFE", BLUE),
        ((130, 440, 820, 560), "预测下一状态、奖励、终止\nz(t+1), r(t), d(t) 的预测值", "#FFEDD5", ORANGE),
        ((130, 635, 820, 755), "与真实结果比较\nL_dyn + L_reward-pred + L_done + L_KL", "#FEE2E2", RED),
        ((130, 830, 820, 950), "更新世界模型参数\nθ、编码器与各预测头", "#EDE9FE", PURPLE),
    ]
    for box, txt, fill, outline in left:
        rounded(d, box, fill, outline); centered(d, box, txt, font(28, True))
    for i in range(len(left)-1):
        arrow(d, (475, left[i][0][3]), (475, left[i+1][0][1]))
    d.text((475, 1050), "目标：预测得准，而不是奖励一定高", font=font(29, True), fill=RED, anchor="mm")

    right = [
        ((1080, 245, 1770, 365), "从真实观察得到当前状态 z_t", "#DBEAFE", BLUE),
        ((1080, 440, 1770, 560), "Actor / Planner 提出候选动作序列\nA^(1), A^(2), …", "#DCFCE7", GREEN),
        ((1080, 635, 1770, 755), "世界模型展开 H 步想象未来\n并预测每一步奖励", "#FFEDD5", ORANGE),
        ((1080, 830, 1770, 950), "比较累计回报，选择最优序列\n只执行第一个动作", "#EDE9FE", PURPLE),
    ]
    for box, txt, fill, outline in right:
        rounded(d, box, fill, outline); centered(d, box, txt, font(28, True))
    for i in range(len(right)-1):
        arrow(d, (1425, right[i][0][3]), (1425, right[i+1][0][1]))
    arrow(d, (1770, 890), (1820, 890), color=GREEN)
    d.line((1820, 890, 1820, 300), fill=GREEN, width=5)
    arrow(d, (1820, 300), (1770, 300), color=GREEN, label="重新观察并规划", label_offset=(-85, -30))
    d.text((1425, 1050), "目标：让长期累计回报尽可能高", font=font(29, True), fill=GREEN, anchor="mm")
    im.save(OUT / "training-vs-decision.png", quality=95)


def rollout_error():
    w, h = 1900, 1080
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    title(d, "多步预测中的分支、不确定性与误差累积", "动作分支由决策产生；随机分支来自环境；滚动误差来自使用预测状态继续预测", w)

    nodes = [
        (190, 480, "当前状态\nz_t", BLUE),
        (530, 280, "动作 A\n预测状态 A", GREEN),
        (530, 680, "动作 B\n预测状态 B", GREEN),
        (920, 190, "可能未来 A1", ORANGE),
        (920, 370, "可能未来 A2", ORANGE),
        (920, 610, "预测第 2 步", ORANGE),
        (1310, 610, "预测第 3 步", RED),
        (1680, 610, "预测第 H 步", RED),
    ]
    for x, y, txt, col in nodes:
        box=(x-120,y-65,x+120,y+65)
        rounded(d, box, WHITE, col, radius=20, width=4)
        centered(d, box, txt, font(26, True))
    arrow(d, (310, 450), (410, 320), color=GREEN, label="候选动作")
    arrow(d, (310, 510), (410, 650), color=GREEN)
    arrow(d, (650, 265), (800, 205), color=ORANGE, label="同一动作的随机未来", label_offset=(0,-45))
    arrow(d, (650, 295), (800, 355), color=ORANGE)
    arrow(d, (650, 680), (800, 625), color=ORANGE)
    arrow(d, (1040, 610), (1190, 610), color=RED)
    arrow(d, (1430, 610), (1560, 610), color=RED)

    d.line((790, 800, 1720, 800), fill=BORDER, width=3)
    d.text((790, 840), "误差规模", font=font(26), fill=MUTED, anchor="ra")
    points=[(820,900),(1060,880),(1320,830),(1580,740),(1740,660)]
    d.line(points, fill=RED, width=7)
    for p in points: d.ellipse((p[0]-8,p[1]-8,p[0]+8,p[1]+8), fill=RED)
    d.text((1260, 970), "模型不断把预测状态当作下一步输入，越往后通常越不可靠", font=font(29, True), fill=RED, anchor="mm")
    d.text((260, 930), "MPC：只执行第一步\n获得真实观察后重新规划", font=font(28, True), fill=NAVY, anchor="mm", spacing=10)
    im.save(OUT / "rollout-error.png", quality=95)


architecture()
training_decision()
rollout_error()
print(OUT)
