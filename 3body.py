import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pygame


# =========================
# CONFIG
# =========================

INITIAL_WIDTH, INITIAL_HEIGHT = 1600, 950
SIDEBAR_WIDTH = 360
MIN_WINDOW_WIDTH = 1100
MIN_WINDOW_HEIGHT = 700

FPS = 60

BACKGROUND = (8, 10, 20)
SIDEBAR_BG = (18, 22, 34)
TEXT_COLOR = (230, 230, 240)
MUTED_TEXT = (170, 175, 190)
GRID_COLOR = (35, 40, 60)
SELECT_COLOR = (255, 255, 255)

BUTTON_BG = (50, 60, 90)
BUTTON_HOVER = (75, 90, 135)
BUTTON_TEXT = (240, 240, 245)
BUTTON_BORDER = (120, 130, 170)

G = 1.0
DT = 0.002
SUBSTEPS = 5
TIME_SCALE = 1.0

INITIAL_ZOOM = 180.0
MIN_ZOOM = 15.0
MAX_ZOOM = 1200.0

MAX_TRAIL_LENGTH = 7000
SHOW_TRAILS = True

BODY_HIT_EXTRA = 8

POS_STEP = 0.1
VEL_STEP = 0.1
MASS_STEP = 0.1


# Runtime window values
WIDTH = INITIAL_WIDTH
HEIGHT = INITIAL_HEIGHT
SIM_WIDTH = WIDTH - SIDEBAR_WIDTH


# =========================
# DATA
# =========================

@dataclass
class Body:
    name: str
    mass: float
    pos: pygame.Vector2
    vel: pygame.Vector2
    color: Tuple[int, int, int]
    radius: int
    affects_others: bool = True

    def copy(self) -> "Body":
        return Body(
            self.name,
            self.mass,
            self.pos.copy(),
            self.vel.copy(),
            self.color,
            self.radius,
            self.affects_others,
        )


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    action: str


# =========================
# PHYSICS
# =========================

def compute_accelerations(bodies: List[Body]) -> List[pygame.Vector2]:
    accs = [pygame.Vector2(0, 0) for _ in bodies]
    softening = 0.015

    for i, bi in enumerate(bodies):
        total = pygame.Vector2(0, 0)
        for j, bj in enumerate(bodies):
            if i == j:
                continue
            if not bj.affects_others:
                continue

            r = bj.pos - bi.pos
            dist_sq = r.length_squared() + softening * softening
            dist = math.sqrt(dist_sq)
            if dist == 0:
                continue

            total += (G * bj.mass / (dist_sq * dist)) * r
        accs[i] = total

    return accs


def rk4_step(bodies: List[Body], dt: float) -> None:
    p0 = [b.pos.copy() for b in bodies]
    v0 = [b.vel.copy() for b in bodies]

    def make_state(ps, vs):
        return [
            Body(
                b.name,
                b.mass,
                ps[i].copy(),
                vs[i].copy(),
                b.color,
                b.radius,
                b.affects_others,
            )
            for i, b in enumerate(bodies)
        ]

    a1 = compute_accelerations(make_state(p0, v0))
    k1p = [v.copy() for v in v0]
    k1v = [a.copy() for a in a1]

    p2 = [p0[i] + 0.5 * dt * k1p[i] for i in range(len(bodies))]
    v2 = [v0[i] + 0.5 * dt * k1v[i] for i in range(len(bodies))]
    a2 = compute_accelerations(make_state(p2, v2))
    k2p = [v.copy() for v in v2]
    k2v = [a.copy() for a in a2]

    p3 = [p0[i] + 0.5 * dt * k2p[i] for i in range(len(bodies))]
    v3 = [v0[i] + 0.5 * dt * k2v[i] for i in range(len(bodies))]
    a3 = compute_accelerations(make_state(p3, v3))
    k3p = [v.copy() for v in v3]
    k3v = [a.copy() for a in a3]

    p4 = [p0[i] + dt * k3p[i] for i in range(len(bodies))]
    v4 = [v0[i] + dt * k3v[i] for i in range(len(bodies))]
    a4 = compute_accelerations(make_state(p4, v4))
    k4p = [v.copy() for v in v4]
    k4v = [a.copy() for a in a4]

    for i, b in enumerate(bodies):
        b.pos = p0[i] + (dt / 6.0) * (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i])
        b.vel = v0[i] + (dt / 6.0) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i])


# =========================
# PRESET / HELPERS
# =========================

def create_default_system() -> List[Body]:
    return [
        Body("m1", 1.0, pygame.Vector2(-1.2, 0.0), pygame.Vector2(0.18, -0.05), (255, 120, 120), 11, True),
        Body("m2", 1.0, pygame.Vector2(1.2, 0.0), pygame.Vector2(-0.18, -0.05), (120, 255, 160), 10, True),
        Body("m3", 2.0, pygame.Vector2(0.0, 1.2), pygame.Vector2(0.0, 0.12), (120, 180, 255), 12, True),
        Body("planet", 0.001, pygame.Vector2(0.0, -0.2), pygame.Vector2(0.9, 0.55), (255, 240, 120), 5, False),
    ]


def copy_bodies(bodies: List[Body]) -> List[Body]:
    return [b.copy() for b in bodies]


def make_trails(bodies: List[Body]):
    return [(b.color, [b.pos.copy()]) for b in bodies]


def make_new_body(index: int, spawn_pos: pygame.Vector2) -> Body:
    colors = [
        (255, 120, 120),
        (120, 255, 160),
        (120, 180, 255),
        (255, 220, 120),
        (220, 120, 255),
        (120, 255, 255),
    ]
    color = colors[index % len(colors)]
    return Body(
        name=f"b{index}",
        mass=1.0,
        pos=spawn_pos.copy(),
        vel=pygame.Vector2(0.0, 0.0),
        color=color,
        radius=10,
        affects_others=True,
    )


# =========================
# CAMERA / COORDS
# =========================

def compute_com(bodies: List[Body]) -> pygame.Vector2:
    total_mass = sum(b.mass for b in bodies if b.affects_others)
    if total_mass == 0:
        return pygame.Vector2(0, 0)

    com = pygame.Vector2(0, 0)
    for b in bodies:
        if b.affects_others:
            com += b.pos * b.mass
    return com / total_mass


def world_to_screen(pos: pygame.Vector2, camera_center: pygame.Vector2, zoom: float) -> Tuple[int, int]:
    x = SIM_WIDTH // 2 + int((pos.x - camera_center.x) * zoom)
    y = HEIGHT // 2 - int((pos.y - camera_center.y) * zoom)
    return x, y


def screen_to_world(screen_pos: Tuple[int, int], camera_center: pygame.Vector2, zoom: float) -> pygame.Vector2:
    sx, sy = screen_pos
    wx = camera_center.x + (sx - SIM_WIDTH // 2) / zoom
    wy = camera_center.y - (sy - HEIGHT // 2) / zoom
    return pygame.Vector2(wx, wy)


# =========================
# UI HELPERS
# =========================

def draw_text(surface, font, text, color, x, y):
    surface.blit(font.render(text, True, color), (x, y))


def make_buttons() -> List[Button]:
    x = SIM_WIDTH + 20
    y = 20
    gap = 12
    w = (SIDEBAR_WIDTH - 3 * 20) // 2
    h = 42

    rows = [
        [("Play", "play"), ("Pause", "pause")],
        [("Edit", "edit"), ("Reset", "reset")],
        [("Trails", "trails"), ("Follow", "follow")],
        [("Add", "add_body"), ("Remove", "remove_body")],
        [("Toggle Tiny", "toggle_tiny")],
    ]

    buttons = []
    current_y = y

    for row in rows:
        if len(row) == 2:
            for col, (label, action) in enumerate(row):
                rect = pygame.Rect(x + col * (w + 20), current_y, w, h)
                buttons.append(Button(label, rect, action))
        else:
            label, action = row[0]
            rect = pygame.Rect(x, current_y, SIDEBAR_WIDTH - 40, h)
            buttons.append(Button(label, rect, action))
        current_y += h + gap

    return buttons


def make_param_controls(start_y: int):
    controls = []
    x0 = SIM_WIDTH + 20
    label_x = x0
    minus_x = x0 + 155
    plus_x = x0 + 255
    row_h = 50

    fields = ["mass", "x", "y", "vx", "vy"]
    for i, field in enumerate(fields):
        y = start_y + i * row_h
        minus_rect = pygame.Rect(minus_x, y, 40, 32)
        plus_rect = pygame.Rect(plus_x, y, 40, 32)
        controls.append((field, label_x, y, minus_rect, plus_rect))

    return controls


def point_in_sim_area(pos):
    return pos[0] < SIM_WIDTH


# =========================
# DRAWING
# =========================

def draw_grid(screen, camera_center, zoom):
    target_px = 100
    spacing_world = target_px / zoom

    base = 10 ** math.floor(math.log10(max(spacing_world, 1e-9)))
    n = spacing_world / base
    if n < 2:
        spacing_world = 1 * base
    elif n < 5:
        spacing_world = 2 * base
    else:
        spacing_world = 5 * base

    left = camera_center.x - SIM_WIDTH / (2 * zoom)
    right = camera_center.x + SIM_WIDTH / (2 * zoom)
    bottom = camera_center.y - HEIGHT / (2 * zoom)
    top = camera_center.y + HEIGHT / (2 * zoom)

    x = math.floor(left / spacing_world) * spacing_world
    while x <= right:
        sx1, sy1 = world_to_screen(pygame.Vector2(x, bottom), camera_center, zoom)
        sx2, sy2 = world_to_screen(pygame.Vector2(x, top), camera_center, zoom)
        pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)
        x += spacing_world

    y = math.floor(bottom / spacing_world) * spacing_world
    while y <= top:
        sx1, sy1 = world_to_screen(pygame.Vector2(left, y), camera_center, zoom)
        sx2, sy2 = world_to_screen(pygame.Vector2(right, y), camera_center, zoom)
        pygame.draw.line(screen, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)
        y += spacing_world


def draw_trails(screen, trails, camera_center, zoom):
    for color, pts in trails:
        if len(pts) < 2:
            continue
        screen_pts = [world_to_screen(p, camera_center, zoom) for p in pts]
        pygame.draw.lines(screen, color, False, screen_pts, 2)


def draw_velocity_arrow(screen, body, camera_center, zoom):
    start = pygame.Vector2(world_to_screen(body.pos, camera_center, zoom))
    end_world = body.pos + body.vel * 0.5
    end = pygame.Vector2(world_to_screen(end_world, camera_center, zoom))
    pygame.draw.line(screen, TEXT_COLOR, start, end, 2)

    d = end - start
    if d.length() > 0:
        d = d.normalize()
        left = end - 10 * d + 5 * pygame.Vector2(-d.y, d.x)
        right = end - 10 * d + 5 * pygame.Vector2(d.y, -d.x)
        pygame.draw.polygon(screen, TEXT_COLOR, [end, left, right])


def draw_bodies(screen, bodies, camera_center, zoom, font, selected_idx):
    for i, b in enumerate(bodies):
        sx, sy = world_to_screen(b.pos, camera_center, zoom)
        pygame.draw.circle(screen, b.color, (sx, sy), b.radius)

        if i == selected_idx:
            pygame.draw.circle(screen, SELECT_COLOR, (sx, sy), b.radius + 4, 2)
            draw_velocity_arrow(screen, b, camera_center, zoom)

        label = font.render(b.name, True, TEXT_COLOR)
        screen.blit(label, (sx + b.radius + 6, sy - b.radius - 6))


def draw_button(screen, font, button):
    mouse_pos = pygame.mouse.get_pos()
    color = BUTTON_HOVER if button.rect.collidepoint(mouse_pos) else BUTTON_BG
    pygame.draw.rect(screen, color, button.rect, border_radius=8)
    pygame.draw.rect(screen, BUTTON_BORDER, button.rect, 2, border_radius=8)
    label = font.render(button.label, True, BUTTON_TEXT)
    label_rect = label.get_rect(center=button.rect.center)
    screen.blit(label, label_rect)


def draw_sidebar(screen, font, small_font, buttons, selected_body, mode, paused, follow_com):
    pygame.draw.rect(screen, SIDEBAR_BG, (SIM_WIDTH, 0, SIDEBAR_WIDTH, HEIGHT))
    pygame.draw.line(screen, (70, 80, 110), (SIM_WIDTH, 0), (SIM_WIDTH, HEIGHT), 2)

    for button in buttons:
        draw_button(screen, small_font, button)

    header_y = 280
    draw_text(screen, font, "Three-Body Sandbox", TEXT_COLOR, SIM_WIDTH + 20, header_y)
    draw_text(screen, small_font, f"Mode: {mode.upper()}", MUTED_TEXT, SIM_WIDTH + 20, header_y + 38)
    draw_text(screen, small_font, f"Paused: {paused}", MUTED_TEXT, SIM_WIDTH + 20, header_y + 62)
    draw_text(screen, small_font, f"Follow COM: {follow_com}", MUTED_TEXT, SIM_WIDTH + 20, header_y + 86)

    y0 = header_y + 130
    draw_text(screen, font, "Selected Body", TEXT_COLOR, SIM_WIDTH + 20, y0)

    if selected_body is None:
        draw_text(screen, small_font, "Click a body to edit it.", MUTED_TEXT, SIM_WIDTH + 20, y0 + 40)
        return

    draw_text(screen, small_font, f"name: {selected_body.name}", TEXT_COLOR, SIM_WIDTH + 20, y0 + 40)
    draw_text(screen, small_font, f"tiny mass body: {not selected_body.affects_others}", MUTED_TEXT, SIM_WIDTH + 20, y0 + 66)

    controls = make_param_controls(y0 + 110)
    values = {
        "mass": selected_body.mass,
        "x": selected_body.pos.x,
        "y": selected_body.pos.y,
        "vx": selected_body.vel.x,
        "vy": selected_body.vel.y,
    }

    for field, label_x, y, minus_rect, plus_rect in controls:
        draw_text(screen, small_font, f"{field}: {values[field]: .4f}", TEXT_COLOR, label_x, y + 4)

        for rect, txt in ((minus_rect, "-"), (plus_rect, "+")):
            pygame.draw.rect(screen, BUTTON_BG, rect, border_radius=6)
            pygame.draw.rect(screen, BUTTON_BORDER, rect, 2, border_radius=6)
            surf = small_font.render(txt, True, BUTTON_TEXT)
            surf_rect = surf.get_rect(center=rect.center)
            screen.blit(surf, surf_rect)

    help_y = y0 + 110 + 5 * 50 + 35
    help_lines = [
        "Edit mode:",
        "  left-drag body to move",
        "  +/- changes use steps of 0.1",
        "  Add / Remove bodies",
        "  Toggle Tiny changes gravity role",
        "  wheel: zoom | right-drag: pan",
        "",
        "Run mode:",
        "  Play starts simulation",
        "  Pause stops motion",
        "  Up/Down change time scale",
    ]
    for i, line in enumerate(help_lines):
        draw_text(screen, small_font, line, MUTED_TEXT, SIM_WIDTH + 20, help_y + 24 * i)


# =========================
# INTERACTION
# =========================

def find_body_at_mouse(bodies, mouse_pos, camera_center, zoom) -> Optional[int]:
    mx, my = mouse_pos
    best = None
    best_d = float("inf")

    for i, b in enumerate(bodies):
        sx, sy = world_to_screen(b.pos, camera_center, zoom)
        d = math.hypot(mx - sx, my - sy)
        if d <= b.radius + BODY_HIT_EXTRA and d < best_d:
            best = i
            best_d = d

    return best


def adjust_body_param(body: Body, field: str, delta: float):
    if field == "mass":
        body.mass = max(0.0001, body.mass + delta)
    elif field == "x":
        body.pos.x += delta
    elif field == "y":
        body.pos.y += delta
    elif field == "vx":
        body.vel.x += delta
    elif field == "vy":
        body.vel.y += delta


def handle_sidebar_click(mouse_pos, buttons, selected_body, control_start_y):
    for button in buttons:
        if button.rect.collidepoint(mouse_pos):
            return ("button", button.action, None)

    if selected_body is None:
        return (None, None, None)

    controls = make_param_controls(control_start_y)
    for field, _, _, minus_rect, plus_rect in controls:
        if minus_rect.collidepoint(mouse_pos):
            return ("param", field, -1)
        if plus_rect.collidepoint(mouse_pos):
            return ("param", field, +1)

    return (None, None, None)


# =========================
# MAIN
# =========================

def main():
    global SHOW_TRAILS, TIME_SCALE, WIDTH, HEIGHT, SIM_WIDTH

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Three-Body Sandbox")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 24)
    small_font = pygame.font.SysFont("consolas", 20)

    edit_bodies = create_default_system()
    sim_initial_bodies = copy_bodies(edit_bodies)
    bodies = copy_bodies(edit_bodies)
    trails = make_trails(bodies)

    mode = "edit"
    paused = True
    follow_com = True
    zoom = INITIAL_ZOOM
    camera_center = compute_com(edit_bodies)

    buttons = make_buttons()

    selected_idx: Optional[int] = 0
    dragging_body_idx: Optional[int] = None
    dragging_camera = False
    drag_offset = pygame.Vector2(0, 0)
    last_mouse = (0, 0)

    while True:
        clock.tick(FPS)

        active_bodies = edit_bodies if mode == "edit" else bodies
        if follow_com:
            camera_center = compute_com(active_bodies)

        header_y = 280
        selected_section_y = header_y + 130
        param_start_y = selected_section_y + 110

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.VIDEORESIZE:
                WIDTH = max(MIN_WINDOW_WIDTH, event.w)
                HEIGHT = max(MIN_WINDOW_HEIGHT, event.h)
                SIM_WIDTH = WIDTH - SIDEBAR_WIDTH
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                buttons = make_buttons()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if point_in_sim_area(event.pos):
                        idx = find_body_at_mouse(active_bodies, event.pos, camera_center, zoom)
                        if idx is not None:
                            selected_idx = idx
                            if mode == "edit":
                                dragging_body_idx = idx
                                mouse_world = screen_to_world(event.pos, camera_center, zoom)
                                drag_offset = active_bodies[idx].pos - mouse_world
                        else:
                            selected_idx = None
                    else:
                        selected_body = active_bodies[selected_idx] if selected_idx is not None and selected_idx < len(active_bodies) else None
                        kind, key, sign = handle_sidebar_click(event.pos, buttons, selected_body, param_start_y)

                        if kind == "button":
                            if key == "play":
                                if mode == "edit":
                                    sim_initial_bodies = copy_bodies(edit_bodies)
                                    bodies = copy_bodies(sim_initial_bodies)
                                    trails = make_trails(bodies)
                                    mode = "run"
                                    paused = False
                                else:
                                    paused = False

                            elif key == "pause":
                                paused = True

                            elif key == "edit":
                                mode = "edit"
                                paused = True
                                edit_bodies = copy_bodies(sim_initial_bodies)

                            elif key == "reset":
                                edit_bodies = create_default_system()
                                sim_initial_bodies = copy_bodies(edit_bodies)
                                bodies = copy_bodies(edit_bodies)
                                trails = make_trails(bodies)
                                mode = "edit"
                                paused = True
                                selected_idx = 0
                                camera_center = compute_com(edit_bodies)

                            elif key == "trails":
                                SHOW_TRAILS = not SHOW_TRAILS

                            elif key == "follow":
                                follow_com = not follow_com

                            elif key == "add_body":
                                if mode == "edit":
                                    new_body = make_new_body(len(edit_bodies) + 1, camera_center)
                                    edit_bodies.append(new_body)
                                    selected_idx = len(edit_bodies) - 1

                            elif key == "remove_body":
                                if mode == "edit" and selected_idx is not None and len(edit_bodies) > 1:
                                    del edit_bodies[selected_idx]
                                    selected_idx = min(selected_idx, len(edit_bodies) - 1) if edit_bodies else None

                            elif key == "toggle_tiny":
                                if mode == "edit" and selected_body is not None:
                                    selected_body.affects_others = not selected_body.affects_others

                        elif kind == "param" and selected_body is not None:
                            step_map = {
                                "mass": MASS_STEP,
                                "x": POS_STEP,
                                "y": POS_STEP,
                                "vx": VEL_STEP,
                                "vy": VEL_STEP,
                            }
                            adjust_body_param(selected_body, key, sign * step_map[key])

                elif event.button == 3:
                    if point_in_sim_area(event.pos):
                        dragging_camera = True
                        last_mouse = event.pos
                        follow_com = False

                elif event.button == 4:
                    if point_in_sim_area(event.pos):
                        old_world = screen_to_world(event.pos, camera_center, zoom)
                        zoom = min(MAX_ZOOM, zoom * 1.1)
                        new_world = screen_to_world(event.pos, camera_center, zoom)
                        camera_center += old_world - new_world

                elif event.button == 5:
                    if point_in_sim_area(event.pos):
                        old_world = screen_to_world(event.pos, camera_center, zoom)
                        zoom = max(MIN_ZOOM, zoom / 1.1)
                        new_world = screen_to_world(event.pos, camera_center, zoom)
                        camera_center += old_world - new_world

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_body_idx = None
                elif event.button == 3:
                    dragging_camera = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging_body_idx is not None and mode == "edit":
                    mouse_world = screen_to_world(event.pos, camera_center, zoom)
                    edit_bodies[dragging_body_idx].pos = mouse_world + drag_offset
                elif dragging_camera:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    camera_center -= pygame.Vector2(dx / zoom, -dy / zoom)
                    last_mouse = event.pos

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                pos_step = POS_STEP * (5 if (mods & pygame.KMOD_SHIFT) else 1)
                vel_step = VEL_STEP * (5 if (mods & pygame.KMOD_SHIFT) else 1)

                if event.key == pygame.K_SPACE and mode == "run":
                    paused = not paused

                elif event.key == pygame.K_RETURN and mode == "edit":
                    sim_initial_bodies = copy_bodies(edit_bodies)
                    bodies = copy_bodies(sim_initial_bodies)
                    trails = make_trails(bodies)
                    mode = "run"
                    paused = False

                elif event.key == pygame.K_e:
                    mode = "edit"
                    paused = True
                    edit_bodies = copy_bodies(sim_initial_bodies)

                elif event.key == pygame.K_r:
                    edit_bodies = create_default_system()
                    sim_initial_bodies = copy_bodies(edit_bodies)
                    bodies = copy_bodies(edit_bodies)
                    trails = make_trails(bodies)
                    mode = "edit"
                    paused = True
                    selected_idx = 0

                elif event.key == pygame.K_t:
                    SHOW_TRAILS = not SHOW_TRAILS

                elif event.key == pygame.K_f:
                    follow_com = not follow_com

                elif event.key == pygame.K_p:
                    if mode == "edit" and selected_idx is not None and selected_idx < len(active_bodies):
                        active_bodies[selected_idx].affects_others = not active_bodies[selected_idx].affects_others

                elif event.key == pygame.K_n and mode == "edit":
                    new_body = make_new_body(len(edit_bodies) + 1, camera_center)
                    edit_bodies.append(new_body)
                    selected_idx = len(edit_bodies) - 1

                elif event.key == pygame.K_DELETE and mode == "edit":
                    if selected_idx is not None and len(edit_bodies) > 1:
                        del edit_bodies[selected_idx]
                        selected_idx = min(selected_idx, len(edit_bodies) - 1) if edit_bodies else None

                elif event.key == pygame.K_UP and mode == "run":
                    TIME_SCALE = min(1000.0, TIME_SCALE * 2.0)

                elif event.key == pygame.K_DOWN and mode == "run":
                    TIME_SCALE = max(0.01, TIME_SCALE / 2.0)

                if selected_idx is not None and selected_idx < len(active_bodies) and mode == "edit":
                    b = active_bodies[selected_idx]
                    if event.key == pygame.K_LEFT:
                        b.pos.x -= pos_step
                    elif event.key == pygame.K_RIGHT:
                        b.pos.x += pos_step
                    elif event.key == pygame.K_DOWN:
                        b.pos.y -= pos_step
                    elif event.key == pygame.K_UP:
                        b.pos.y += pos_step
                    elif event.key == pygame.K_a:
                        b.vel.x -= vel_step
                    elif event.key == pygame.K_d:
                        b.vel.x += vel_step
                    elif event.key == pygame.K_w:
                        b.vel.y += vel_step
                    elif event.key == pygame.K_s:
                        b.vel.y -= vel_step
                    elif event.key == pygame.K_0:
                        b.vel = pygame.Vector2(0, 0)

        if mode == "run" and not paused:
            effective_dt = DT * TIME_SCALE
            for _ in range(SUBSTEPS):
                rk4_step(bodies, effective_dt / SUBSTEPS)

            for i, b in enumerate(bodies):
                trails[i][1].append(b.pos.copy())
                if len(trails[i][1]) > MAX_TRAIL_LENGTH:
                    trails[i][1].pop(0)

        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, BACKGROUND, (0, 0, SIM_WIDTH, HEIGHT))
        draw_grid(screen, camera_center, zoom)

        if SHOW_TRAILS:
            draw_trails(screen, trails, camera_center, zoom)

        draw_bodies(screen, active_bodies, camera_center, zoom, small_font, selected_idx)

        selected_body = None
        if selected_idx is not None and selected_idx < len(active_bodies):
            selected_body = active_bodies[selected_idx]

        draw_sidebar(screen, font, small_font, buttons, selected_body, mode, paused, follow_com)

        draw_text(screen, small_font, f"zoom: {zoom:.1f}", MUTED_TEXT, 20, 20)
        draw_text(screen, small_font, f"time scale: {TIME_SCALE:.2f}", MUTED_TEXT, 20, 45)
        draw_text(screen, small_font, "Left drag: move body | Right drag: pan | Wheel: zoom", MUTED_TEXT, 20, 70)

        pygame.display.flip()


if __name__ == "__main__":
    main()