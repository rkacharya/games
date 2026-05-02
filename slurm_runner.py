"""
SLURM Runner — endless runner game, cartridge edition.

Loaded and exec()'d at runtime by the launcher. Exposes:
  fetch_slurm_stats() -> SlurmStats
  SlurmRunner(stdscr, stats).run()

Controls:
  Space / Up / w    Jump
  Down / s          Duck (avoid overhead MEM obstacles)
  q / Esc           Quit

Obstacles:
  ERR  — failed job block (2 rows) — jump over
  OOM  — out-of-memory (3 rows)   — big jump
  MEM  — memory hog (overhead)    — duck under
  NET  — network clog (1 row)     — jump or slow-walk through
"""

import curses
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Size requirements
# ---------------------------------------------------------------------------
MIN_W = 80
MIN_H = 22

# ---------------------------------------------------------------------------
# Gameplay constants
# ---------------------------------------------------------------------------
PLAYER_COL    = 10     # fixed screen col offset from sx0
LIVES_START   = 3
INVULN_TICKS  = 30     # ~3 s at 10 fps
CRUMBLE_TICKS = 15     # ticks before crumble tile becomes a pit

BASE_SPEED    = 0.4    # cols-per-tick at 0% cluster load
MAX_SPEED     = 1.8    # cols-per-tick at 100% cluster load
SPEED_CREEP   = 0.002  # speed increases by this every tick

# Jump arc: integer row offsets from ground each tick (negative = above ground).
# Player feet move to ground_row + JUMP_ARC[jump_tick] each tick while airborne.
JUMP_ARC = [0, -1, -2, -3, -4, -4, -3, -3, -2, -1, 0]

MIN_OBS_GAP   = 22     # minimum cols between obstacle right edge and next obstacle
GRACE_COLS    = 35     # no obstacles in the first N cols

# ---------------------------------------------------------------------------
# Color pair IDs
# ---------------------------------------------------------------------------
C_CAB   = 1   # TV cabinet — white
C_GLASS = 2   # screen glass border — cyan
C_SKY   = 3   # stars — white dim
C_GND   = 4   # ground tiles — green bold
C_CRUMB = 5   # crumble tiles — yellow bold
C_PLR   = 6   # player — magenta bold
C_OBS   = 7   # obstacles — red bold
C_COIN  = 8   # collectibles — yellow bold
C_HUD   = 9   # HUD bar — white on blue
C_DIM   = 10  # footer / dim text — white dim
C_INV   = 11  # invincible player — cyan bold
C_NET   = 12  # NET obstacle — cyan bold
# Rainbow pairs for has_job player (cycle through these)
C_R1    = 13  # red
C_R2    = 14  # green
C_R3    = 15  # yellow
C_R4    = 16  # blue
C_R5    = 17  # magenta
C_R6    = 18  # cyan
C_R7    = 19  # white
RAINBOW = [C_R1, C_R2, C_R3, C_R4, C_R5, C_R6, C_R7]


# ---------------------------------------------------------------------------
# SLURM statistics
# ---------------------------------------------------------------------------

@dataclass
class SlurmStats:
    cpu_alloc:      int   = 0
    cpu_total:      int   = 1
    cpu_pct:        float = 0.5
    jobs_running:   int   = 0
    jobs_pending:   int   = 100
    nodes_draining: int   = 0
    nodes_down:     int   = 0
    nodes_total:    int   = 1
    user_jobs:      int        = 0
    job_names:      List[str]  = field(default_factory=list)


def fetch_slurm_stats() -> SlurmStats:
    s = SlurmStats()
    try:
        r = subprocess.run(
            ['sinfo', '--noheader', '-o', '%C'],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split('/')
            if len(parts) == 4:
                s.cpu_alloc = int(parts[0])
                s.cpu_total = max(1, int(parts[3]))
                s.cpu_pct   = s.cpu_alloc / s.cpu_total

        r2 = subprocess.run(
            ['sinfo', '--noheader', '-o', '%t'],
            capture_output=True, text=True, timeout=3
        )
        if r2.returncode == 0:
            lines = r2.stdout.strip().splitlines()
            s.nodes_total    = len(lines)
            s.nodes_draining = sum(1 for l in lines if l.strip() in ('drng', 'drain'))
            s.nodes_down     = sum(1 for l in lines if l.strip() in ('down*', 'down', 'inval'))

        r3 = subprocess.run(
            ['squeue', '--noheader', '-o', '%T'],
            capture_output=True, text=True, timeout=3
        )
        if r3.returncode == 0:
            states = [l.strip() for l in r3.stdout.strip().splitlines()]
            s.jobs_running = states.count('RUNNING')
            s.jobs_pending = states.count('PENDING')

        user = os.environ.get('USER', '')
        if user:
            r4 = subprocess.run(
                ['squeue', '--noheader', '-u', user, '-o', '%j'],
                capture_output=True, text=True, timeout=3
            )
            if r4.returncode == 0:
                names = [l.strip() for l in r4.stdout.strip().splitlines() if l.strip()]
                s.user_jobs  = len(names)
                s.job_names  = names
    except Exception:
        pass
    return s


def _difficulty(pct: float) -> str:
    if pct < 0.25: return 'QUIET'
    if pct < 0.50: return 'MEDIUM'
    if pct < 0.75: return 'HEAVY'
    return 'EXTREME'


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """
    Screen-coordinate obstacle.
    col: screen col relative to sx0 (the left edge of the play area).
    kind: 'err' | 'oom' | 'mem' | 'net'
    width: in screen cols
    height: in screen rows above ground_row (for ground obstacles)
    overhead: True for MEM — draws 2 rows above ground_row; player must duck
    """
    col:      int
    kind:     str
    width:    int
    height:   int   = 1
    overhead: bool  = False

    def rows(self, ground_row: int) -> List[Tuple[int, str]]:
        """Return list of (screen_row, chars) pairs, drawn top-to-bottom."""
        w = self.width
        if self.kind == 'err':
            return [
                (ground_row - 1, '▀' * w),
                (ground_row,     '█' * w),
            ]
        if self.kind == 'oom':
            return [
                (ground_row - 2, '▀' * w),
                (ground_row - 1, '█' * w),
                (ground_row,     '█' * w),
            ]
        if self.kind == 'mem':
            # overhead: draws 2 rows above ground_row
            return [(ground_row - 2, '▄' * w)]
        if self.kind == 'net':
            return [(ground_row, '≋' * w)]
        return [(ground_row, '?' * w)]


# ---------------------------------------------------------------------------
# Collectible
# ---------------------------------------------------------------------------

@dataclass
class Collectible:
    col:      int
    row_off:  int    # row = ground_row + row_off (negative = above ground)
    kind:     str    # 'cpu' | 'node'

    def char(self) -> str:
        return '$' if self.kind == 'cpu' else '★'


# ---------------------------------------------------------------------------
# Ground tile
# ---------------------------------------------------------------------------

@dataclass
class GroundTile:
    crumble:       bool = False
    crumble_timer: int  = -1   # -1 = not triggered
    pit:           bool = False

    @property
    def char(self) -> str:
        if self.pit:           return ' '
        if not self.crumble:   return '▓'
        if self.crumble_timer < 0:  return '░'
        if self.crumble_timer < 5:  return '·'
        return '░'

    @property
    def solid(self) -> bool:
        return not self.pit and not (self.crumble and self.crumble_timer >= CRUMBLE_TICKS)


# ---------------------------------------------------------------------------
# Main game
# ---------------------------------------------------------------------------

class SlurmRunner:

    def __init__(self, stdscr, stats: SlurmStats):
        self.stdscr = stdscr
        self.stats  = stats
        self.h, self.w = stdscr.getmaxyx()
        self._rng = random.Random()
        self._compute_layout()
        self._reset()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _compute_layout(self):
        h, w = self.h, self.w

        self.ctrl_w    = 13
        self.divider   = w - self.ctrl_w - 1

        self.cab_top    = 2
        self.label_row  = 3
        self.sep_row    = 4
        self.play_top   = 5
        self.play_bot   = h - 4
        self.grille_row = h - 3
        self.cab_bot    = h - 2

        # Screen interior bounds (inside glass corners)
        self.sx0 = 2
        self.sx1 = self.divider - 2
        self.sy0 = self.play_top + 1   # HUD row
        self.sy1 = self.play_bot - 1   # footer row

        self.sw = self.sx1 - self.sx0 + 1   # play width in cols
        self.sh = self.sy1 - self.sy0 + 1   # play height in rows

        # Row assignments inside the glass:
        #   sy0           = HUD bar (top)
        #   sy0+1..gr-1   = sky / play area
        #   ground_row    = player feet / obstacle bases
        #   ground_row+1  = ▓ tile row
        #   sy1           = footer text
        self.ground_row = self.sy1 - 2
        self.play_sy0   = self.sy0 + 1
        self.sky_rows   = max(1, (self.ground_row - self.play_sy0) // 2)

    # ------------------------------------------------------------------
    # Reset / init
    # ------------------------------------------------------------------

    def _reset(self):
        self.tick_n       = 0
        self.score        = 0
        self.distance     = 0.0
        self.scroll_speed = BASE_SPEED + self.stats.cpu_pct * (MAX_SPEED - BASE_SPEED)
        self.scroll_frac  = 0.0
        self.cols_scrolled = 0    # total cols scrolled (for obstacle spacing)
        self.done         = False

        # Player state
        self.lives     = LIVES_START
        self.inv_ticks = 0
        self.air_row   = 0        # 0 = on ground; negative = rows above ground
        self.jump_tick = -1       # -1 = not jumping; >=0 = index into JUMP_ARC
        self.ducking   = False
        self.anim_f    = 0
        self.has_job   = self.stats.user_jobs > 0
        self.fall_row  = 0        # how far below ground player has fallen (pit death)
        self.speed_mul = 1.0
        self._job_name_idx = 0    # cycles through job_names for collect messages

        # Collect message flash
        self.collect_msg       = ''
        self.collect_msg_ticks = 0

        # Ground: dict from screen col (relative to sx0) to GroundTile
        # Populated for all cols 0..sw+LOOKAHEAD
        self.ground: Dict[int, GroundTile] = {}
        self._pits_budget    = min(self.stats.nodes_down, 5)
        self._crumble_budget = min(self.stats.nodes_draining, 10)
        self._obs_gap        = max(18, int(MIN_OBS_GAP * (1.0 - self.stats.cpu_pct * 0.3)))
        self._last_obs_right = 0   # right edge col of the last obstacle spawned
        self._obs_rate       = min(0.85, 0.3 + (self.stats.jobs_pending / 300.0) * 0.55)
        self._last_obs_kind  = ''

        # Generate initial ground
        for c in range(self.sw + 30):
            self.ground[c] = self._new_tile(c)

        self.obstacles:    List[Obstacle]    = []
        self.collectibles: List[Collectible] = []

        # Stars: list of (col_offset, row_offset_from_play_sy0)
        n_stars = max(4, self.sw // 6)
        self.stars: List[Tuple[int, int]] = [
            (self._rng.randint(0, self.sw - 1),
             self._rng.randint(0, max(0, self.sky_rows - 1)))
            for _ in range(n_stars)
        ]
        self.star_tick = 0

    def _new_tile(self, col: int) -> GroundTile:
        t = GroundTile()
        if col < GRACE_COLS:
            return t
        if (self._pits_budget > 0
                and self._rng.random() < 0.025
                and col > 5):
            t.pit = True
            self._pits_budget -= 1
        elif (self._crumble_budget > 0
              and self._rng.random() < 0.05):
            t.crumble = True
            self._crumble_budget -= 1
        return t

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_colors(self):
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(C_CAB,   curses.COLOR_WHITE,   -1)
        curses.init_pair(C_GLASS, curses.COLOR_CYAN,    -1)
        curses.init_pair(C_SKY,   curses.COLOR_WHITE,   -1)
        curses.init_pair(C_GND,   curses.COLOR_GREEN,   -1)
        curses.init_pair(C_CRUMB, curses.COLOR_YELLOW,  -1)
        curses.init_pair(C_PLR,   curses.COLOR_MAGENTA, -1)
        curses.init_pair(C_OBS,   curses.COLOR_RED,     -1)
        curses.init_pair(C_COIN,  curses.COLOR_YELLOW,  -1)
        curses.init_pair(C_HUD,   curses.COLOR_WHITE,   curses.COLOR_BLUE)
        curses.init_pair(C_DIM,   curses.COLOR_WHITE,   -1)
        curses.init_pair(C_INV,   curses.COLOR_CYAN,    -1)
        curses.init_pair(C_NET,   curses.COLOR_CYAN,    -1)
        curses.init_pair(C_R1,    curses.COLOR_RED,     -1)
        curses.init_pair(C_R2,    curses.COLOR_GREEN,   -1)
        curses.init_pair(C_R3,    curses.COLOR_YELLOW,  -1)
        curses.init_pair(C_R4,    curses.COLOR_BLUE,    -1)
        curses.init_pair(C_R5,    curses.COLOR_MAGENTA, -1)
        curses.init_pair(C_R6,    curses.COLOR_CYAN,    -1)
        curses.init_pair(C_R7,    curses.COLOR_WHITE,   -1)

    def _attr(self, pair: int, bold: bool = False, dim: bool = False) -> int:
        a = curses.color_pair(pair) if curses.has_colors() else curses.A_NORMAL
        if bold: a |= curses.A_BOLD
        if dim:  a |= curses.A_DIM
        return a

    def _sa(self, row: int, col: int, s: str, attr: int = 0) -> None:
        """Safe addstr clipped to terminal bounds."""
        if row < 0 or row >= self.h - 1:
            return
        if col >= self.w - 1:
            return
        if col < 0:
            s = s[-col:]
            col = 0
        s = s[:max(0, self.w - 1 - col)]
        if not s:
            return
        try:
            self.stdscr.addstr(row, col, s, attr)
        except curses.error:
            pass

    # ------------------------------------------------------------------
    # Physics / tick
    # ------------------------------------------------------------------

    def _tick(self, duck_held: bool):
        self.tick_n += 1
        self.ducking = duck_held and self.jump_tick < 0 and self.air_row == 0

        # Speed ramp
        self.scroll_speed = min(MAX_SPEED, self.scroll_speed + SPEED_CREEP)

        # Jump arc
        if self.jump_tick >= 0:
            self.jump_tick += 1
            if self.jump_tick >= len(JUMP_ARC):
                self.jump_tick = -1
                self.air_row   = 0
            else:
                self.air_row = JUMP_ARC[self.jump_tick]

        # Pit fall physics
        if self.air_row == 0:
            tile = self.ground.get(PLAYER_COL)
            if tile is None or tile.pit:
                self.fall_row += 1
                if self.fall_row > 3:
                    self._get_hit()
                    self.fall_row = 0
            else:
                self.fall_row = 0
                # Crumble trigger
                if tile.crumble and tile.crumble_timer < 0:
                    tile.crumble_timer = 0

        # Crumble timers
        for t in self.ground.values():
            if t.crumble and t.crumble_timer >= 0:
                t.crumble_timer += 1
                if t.crumble_timer >= CRUMBLE_TICKS:
                    t.pit    = True
                    t.crumble = False

        # Walk animation
        if self.tick_n % 5 == 0 and self.air_row == 0 and not self.ducking:
            self.anim_f = 1 - self.anim_f

        # Invincibility countdown
        if self.inv_ticks > 0:
            self.inv_ticks -= 1

        # Collect message countdown
        if self.collect_msg_ticks > 0:
            self.collect_msg_ticks -= 1

        # NET speed penalty: if player is on ground and inside a NET obstacle
        self.speed_mul = 1.0
        for obs in self.obstacles:
            if obs.kind == 'net' and self.air_row == 0:
                if obs.col <= PLAYER_COL <= obs.col + obs.width - 1:
                    self.speed_mul = 0.6

        # Scroll
        self.scroll_frac += self.scroll_speed * self.speed_mul
        scroll_steps = int(self.scroll_frac)
        self.scroll_frac -= scroll_steps
        for _ in range(scroll_steps):
            self._scroll_one()

        self.score    += 1
        self.distance += self.scroll_speed * self.speed_mul

        # Collision (skip while invincible)
        if self.inv_ticks == 0:
            self._check_collision()

        # Collectibles
        self._check_collectibles()

        if self.lives <= 0:
            self.done = True

    def _scroll_one(self):
        """Advance world by one column."""
        self.cols_scrolled += 1

        # Shift ground dict left
        new_ground: Dict[int, GroundTile] = {}
        for c, t in self.ground.items():
            if c - 1 >= -1:
                new_ground[c - 1] = t
        # Generate new rightmost col
        right = self.sw + 10
        new_ground[right] = self._new_tile(self.cols_scrolled + right)
        self.ground = new_ground

        # Shift obstacles
        for obs in self.obstacles:
            obs.col -= 1
        self.obstacles = [o for o in self.obstacles if o.col + o.width > -2]

        # Shift collectibles
        for col in self.collectibles:
            col.col -= 1
        self.collectibles = [c for c in self.collectibles if c.col > -2]

        # Shift stars
        self.star_tick += 1
        if self.star_tick % 3 == 0:
            self.stars = [((c - 1) % self.sw, r) for c, r in self.stars]

        # Spawn new obstacles
        self._maybe_spawn_obstacle()

    def _maybe_spawn_obstacle(self):
        if self.cols_scrolled < GRACE_COLS:
            return
        # Find rightmost obstacle right edge
        if self.obstacles:
            rightmost = max(o.col + o.width for o in self.obstacles)
        else:
            rightmost = self._last_obs_right

        spawn_col = self.sw + 5   # off-screen right
        gap = spawn_col - rightmost
        if gap < self._obs_gap:
            return
        if self._rng.random() > self._obs_rate / 6:
            return

        kind = self._pick_obs_kind()
        width = self._rng.randint(2, 4)
        height = {'err': 2, 'oom': 3, 'mem': 1, 'net': 1}[kind]
        overhead = (kind == 'mem')
        obs = Obstacle(col=spawn_col, kind=kind, width=width,
                       height=height, overhead=overhead)
        self.obstacles.append(obs)
        self._last_obs_right = spawn_col + width
        self._last_obs_kind  = kind

        # Maybe add a collectible before it
        if self._rng.random() < 0.35:
            cx = spawn_col - self._rng.randint(4, 8)
            ck = 'node' if self._rng.random() < 0.15 else 'cpu'
            row_off = -2 if ck == 'node' else -1
            self.collectibles.append(Collectible(col=cx, row_off=row_off, kind=ck))

    def _pick_obs_kind(self) -> str:
        choices = ['err', 'oom', 'mem', 'net']
        if self._last_obs_kind in choices:
            choices = [c for c in choices if c != self._last_obs_kind]
        return self._rng.choice(choices)

    def _get_hit(self):
        if self.inv_ticks > 0:
            return
        self.lives    -= 1
        self.inv_ticks = INVULN_TICKS
        # Reset to ground
        self.jump_tick = -1
        self.air_row   = 0
        self.ducking   = False

    def _check_collision(self):
        """Precise row/col collision between player and each obstacle."""
        # Player column range on screen (relative to sx0)
        sprite_w = 1 if self.ducking else 3
        p_col_lo = PLAYER_COL
        p_col_hi = PLAYER_COL + sprite_w - 1

        # Player row range: air_row=0 → ground_row; -1 → one above; etc.
        if self.ducking:
            p_row_lo = p_row_hi = self.ground_row
        else:
            p_row_hi = self.ground_row + self.air_row          # bottom (feet)
            p_row_lo = p_row_hi - 2                            # top (head)

        for obs in self.obstacles:
            # Horizontal overlap?
            o_col_lo = obs.col
            o_col_hi = obs.col + obs.width - 1
            if p_col_hi < o_col_lo or p_col_lo > o_col_hi:
                continue

            # Vertical overlap?
            for row, _ in obs.rows(self.ground_row):
                if p_row_lo <= row <= p_row_hi:
                    self._get_hit()
                    return

    def _check_collectibles(self):
        p_col = PLAYER_COL
        p_row = self.ground_row + self.air_row
        for c in list(self.collectibles):
            c_row = self.ground_row + c.row_off
            if abs(c.col - p_col) <= 2 and abs(c_row - p_row) <= 1:
                if c.kind == 'cpu':
                    self.score += 10
                    label = 'cpu_core'
                else:
                    self.score    += 50
                    self.inv_ticks = INVULN_TICKS
                    label = 'node_token'
                # Show a job name from the queue if available
                names = self.stats.job_names
                if names:
                    job = names[self._job_name_idx % len(names)]
                    self._job_name_idx += 1
                    self.collect_msg = f'COLLECTED: {job}'
                else:
                    pts = '+10' if c.kind == 'cpu' else '+50'
                    self.collect_msg = f'COLLECTED: {label} ({pts})'
                self.collect_msg_ticks = 18   # ~1.8 s at 10fps
                self.collectibles.remove(c)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_tv_cabinet(self):
        h, w = self.h, self.w
        cab  = self._attr(C_CAB,   bold=True)
        dim  = self._attr(C_CAB,   dim=True)
        scr  = self._attr(C_GLASS, bold=True)

        # Antennas
        al = w // 4
        ar = w - w // 4
        self._sa(0, al - 2, '/\\', dim)
        self._sa(1, al - 3, '/',   dim)
        self._sa(0, ar,     '/\\', dim)
        self._sa(1, ar + 2, '\\',  dim)

        # Cabinet top
        self._sa(self.cab_top, 0, ('╭' + '─' * (w - 2) + '╮')[:w - 1], cab)

        # Label
        label = 'S  L  U  R  M  -  T  V'
        self._sa(self.label_row, 0, ('│' + label.center(w - 2) + '│')[:w - 1], cab)

        # Separator with divider
        dv  = self.divider
        sep = '├' + '─' * (dv - 1) + '┬' + '─' * (w - dv - 2) + '┤'
        self._sa(self.sep_row, 0, sep[:w - 1], cab)

        # Side borders
        for r in range(self.play_top, self.play_bot + 1):
            self._sa(r, 0,       '│', cab)
            self._sa(r, dv,      '│', cab)
            self._sa(r, w - 1,   '│', cab)

        # Screen glass
        gl, gr = 1, dv - 1
        gt, gb = self.play_top, self.play_bot
        self._sa(gt, gl, '╭' + '─' * (gr - gl - 1) + '╮', scr)
        self._sa(gb, gl, '╰' + '─' * (gr - gl - 1) + '╯', scr)
        for r in range(gt + 1, gb):
            self._sa(r, gl, '│', scr)
            self._sa(r, gr, '│', scr)

        # Scanline background
        for r in range(self.sy0, self.sy1 + 1):
            fill = ' ' * (self.sx1 - self.sx0 + 1)
            self._sa(r, self.sx0, fill, dim if r % 2 == 0 else 0)

        # Control panel knobs
        cx = dv + 2
        ph = self.play_bot - self.play_top
        self._sa(self.play_top + max(1, ph // 5),     cx, 'CH+',   cab)
        self._sa(self.play_top + max(2, ph // 5 + 1), cx, '(   )', cab)
        self._sa(self.play_top + max(3, ph * 2 // 5), cx, 'VOL',   cab)
        self._sa(self.play_top + max(4, ph * 2 // 5 + 1), cx, '(   )', cab)
        self._sa(self.play_top + max(5, ph * 3 // 5), cx, 'PWR',   cab)
        self._sa(self.play_top + max(6, ph * 3 // 5 + 1), cx, '[ ■ ]', cab)

        # Cluster stats
        spd_r = self.play_top + max(7, ph * 4 // 5)
        self._sa(spd_r,     cx, 'SPD', dim)
        self._sa(spd_r + 1, cx, f'{self.scroll_speed:.1f}x', cab)

        nd_tot = max(1, self.stats.nodes_total)
        nd_up  = nd_tot - self.stats.nodes_down - self.stats.nodes_draining
        r = spd_r + 3
        self._sa(r,     cx, 'NODES', dim)
        self._sa(r + 1, cx, f'{nd_up}/{nd_tot}', cab)
        self._sa(r + 3, cx, 'JOBS',  dim)
        self._sa(r + 4, cx, f'{self.stats.jobs_running}R', cab)
        self._sa(r + 5, cx, f'{self.stats.jobs_pending}P', dim)
        pw = 8
        bar = '█' * int(self.stats.cpu_pct * pw) + '░' * (pw - int(self.stats.cpu_pct * pw))
        self._sa(r + 7, cx, 'LOAD', dim)
        self._sa(r + 8, cx, bar, self._attr(C_HUD, bold=True))

        # Bottom grille separator
        gs = '├' + '─' * (dv - 1) + '┴' + '─' * (w - dv - 2) + '┤'
        self._sa(self.grille_row, 0, gs[:w - 1], cab)
        self._sa(self.grille_row + 1, 1, ('░' * (w - 2))[:w - 2], dim)

        # Cabinet bottom
        self._sa(self.cab_bot, 0, ('╰' + '─' * (w - 2) + '╯')[:w - 1], cab)

    def _draw_hud(self):
        hearts = '♥' * self.lives + '♡' * (LIVES_START - self.lives)
        pw = 8
        bar = '█' * int(self.stats.cpu_pct * pw) + '░' * (pw - int(self.stats.cpu_pct * pw))
        nd  = self.stats.nodes_total
        ndn = self.stats.nodes_down
        ndd = self.stats.nodes_draining
        node_info = f'  ↓{ndn} ~{ndd}' if (ndn or ndd) else ''
        inv = ' ★INV' if self.inv_ticks > 0 else ''
        hud = (f' {hearts}  CPU:{bar}{int(self.stats.cpu_pct*100):2d}%'
               f'  R:{self.stats.jobs_running} P:{self.stats.jobs_pending}'
               f'{node_info}'
               f'  {self.score:06d}{inv} ')
        self._sa(self.sy0, self.sx0,
                 hud[:self.sw].ljust(self.sw),
                 self._attr(C_HUD, bold=True))

    def _draw_footer(self):
        dist_km = self.distance / 100.0
        nd = self.stats.nodes_total
        text = (f'  {self.scroll_speed:.1f}x'
                f'  {dist_km:.1f}km'
                f'  {_difficulty(self.stats.cpu_pct)}'
                f'  {nd}nodes'
                f'  ERR=jump OOM=bigjump MEM=duck NET=slow')
        self._sa(self.sy1, self.sx0,
                 text[:self.sw],
                 self._attr(C_DIM, dim=True))

    def _draw_stars(self):
        dim = self._attr(C_SKY, dim=True)
        for sc, sr in self.stars:
            row = self.play_sy0 + sr
            col = self.sx0 + sc
            if self.play_sy0 <= row <= self.ground_row - 1 and self.sx0 <= col <= self.sx1:
                self._sa(row, col, '.', dim)

    def _draw_ground(self):
        tile_row = self.ground_row + 1
        for c in range(self.sw + 2):
            tile = self.ground.get(c)
            screen_col = self.sx0 + c
            if screen_col < self.sx0 or screen_col > self.sx1:
                continue
            if tile is None or tile.pit:
                continue
            attr = (self._attr(C_CRUMB, bold=True) if tile.crumble
                    else self._attr(C_GND, bold=True))
            self._sa(tile_row, screen_col, tile.char, attr)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            if obs.col + obs.width < 0 or obs.col >= self.sw:
                continue
            if obs.kind == 'net':
                attr = self._attr(C_NET, bold=True)
            else:
                attr = self._attr(C_OBS, bold=True)
            for row, chars in obs.rows(self.ground_row):
                if self.play_sy0 <= row <= self.ground_row:
                    col = self.sx0 + obs.col
                    self._sa(row, col, chars[:self.sx1 - col + 1], attr)

    def _draw_collectibles(self):
        attr = self._attr(C_COIN, bold=True)
        for c in self.collectibles:
            row = self.ground_row + c.row_off
            col = self.sx0 + c.col
            if self.play_sy0 <= row <= self.ground_row and self.sx0 <= col <= self.sx1:
                self._sa(row, col, c.char(), attr)

    def _draw_player(self):
        # Blink during invincibility
        if self.inv_ticks > 0 and self.tick_n % 6 < 3:
            return

        # Rainbow cycle when player has jobs running; otherwise fixed magenta
        if self.has_job and self.inv_ticks == 0:
            pair = RAINBOW[(self.tick_n // 3) % len(RAINBOW)]
            attr = self._attr(pair, bold=True)
        elif self.inv_ticks > 0:
            attr = self._attr(C_INV, bold=True)
        else:
            attr = self._attr(C_PLR, bold=True)

        # Single-char sprite matching the TV mini-preview exactly
        if self.ducking:
            char = '['
        elif self.air_row < 0:
            char = '↑'
        elif self.anim_f == 0:
            char = '►'
        else:
            char = '▶'

        row = self.ground_row + self.air_row
        col = self.sx0 + PLAYER_COL
        if self.play_sy0 <= row <= self.ground_row:
            self._sa(row, col, char, attr)

    def _draw_collect_msg(self):
        if not self.collect_msg or self.collect_msg_ticks <= 0:
            return
        msg  = f' {self.collect_msg} '
        row  = self.ground_row - 2
        col  = self.sx0 + PLAYER_COL + 2
        col  = min(col, self.sx1 - len(msg))
        if row >= self.play_sy0:
            self._sa(row, col, msg, self._attr(C_COIN, bold=True))

    def _draw(self):
        self.stdscr.erase()
        self._draw_tv_cabinet()
        self._draw_stars()
        self._draw_ground()
        self._draw_obstacles()
        self._draw_collectibles()
        self._draw_player()
        self._draw_collect_msg()
        self._draw_hud()
        self._draw_footer()
        self.stdscr.refresh()

    # ------------------------------------------------------------------
    # Game over
    # ------------------------------------------------------------------

    def _game_over(self) -> bool:
        self.stdscr.erase()
        self._draw_tv_cabinet()

        dist_km = self.distance / 100.0
        lines = [
            'G A M E   O V E R',
            '',
            f'SCORE:    {self.score:06d}',
            f'DISTANCE: {dist_km:.1f} km',
            f'LOAD:     {_difficulty(self.stats.cpu_pct)}',
            f'NODES:    {self.stats.nodes_total}',
            '',
            '[R] Play again     [Q] Quit',
        ]

        start_row = self.sy0 + max(0, (self.sh - len(lines)) // 2)
        for i, line in enumerate(lines):
            row = start_row + i
            if row > self.sy1:
                break
            col = self.sx0 + max(0, (self.sw - len(line)) // 2)
            if i == 0:
                attr = self._attr(C_OBS, bold=True)
            elif '[R]' in line:
                attr = self._attr(C_HUD, bold=True)
            else:
                attr = self._attr(C_DIM, dim=True)
            self._sa(row, col, line[:self.sx1 - col + 1], attr)

        self.stdscr.refresh()
        curses.nocbreak()
        self.stdscr.keypad(True)
        curses.cbreak()
        while True:
            key = self.stdscr.getch()
            if key in (ord('r'), ord('R')):
                return True
            if key in (ord('q'), ord('Q'), 27):
                return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        curses.curs_set(0)
        self._init_colors()

        while True:
            curses.halfdelay(1)
            duck_held = False

            while not self.done:
                self._draw()
                key = self.stdscr.getch()

                if key == curses.ERR:
                    duck_held = False
                elif key in (27, ord('q')):
                    return
                elif key in (ord(' '), curses.KEY_UP, ord('w')):
                    if self.jump_tick < 0 and self.air_row == 0:
                        self.jump_tick = 0
                        self.air_row   = JUMP_ARC[0]
                    duck_held = False
                elif key in (curses.KEY_DOWN, ord('s')):
                    duck_held = True

                self._tick(duck_held)

            play_again = self._game_over()
            if not play_again:
                return
            self._reset()
