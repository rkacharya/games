"""
SLURM Runner — endless runner game, cartridge edition.

Loaded and exec()'d at runtime by the launcher. Exposes:
  fetch_slurm_stats() -> SlurmStats
  SlurmRunner(stdscr, stats).run()

Controls:
  Space / Up    Jump
  Down / s      Duck (avoid overhead obstacles)
  q / Esc       Quit

Obstacle types:
  [ERR]   Failed job block — jump over (tall)
  ▄▄▄     Memory error — duck under (low, overhead)
  |OOM|   Out-of-memory killer — big jump needed (very tall)
  ≋≋≋     Network congestion — jump over (slows you if landed on)

Collectibles:
  ©       CPU core token  (+10 pts)
  [N]     Node token      (+50 pts, brief invincibility)

The ground itself reflects cluster health:
  ▓       Normal ground tile
  ░       Crumbling tile (draining node) — collapses after you step on it
  (gap)   Pit (down node) — instant death
"""

import curses
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Terminal size requirements
# ---------------------------------------------------------------------------
MIN_W = 80
MIN_H = 22

# ---------------------------------------------------------------------------
# Physics & gameplay constants
# ---------------------------------------------------------------------------
BASE_SPEED     = 0.5   # scroll tiles/tick at 0% cluster load
MAX_SPEED      = 2.2   # scroll tiles/tick at 100% cluster load
SPEED_CREEP    = 0.003 # speed increase per tick (slow ramp regardless of load)
JUMP_V         = 4.0   # initial upward velocity on jump (world units/tick)
GRAVITY        = 0.55  # downward acceleration (world units/tick²)
PLAYER_COL     = 10    # fixed screen column where player sprite sits
LIVES_START    = 3
INVULN_TICKS   = 40    # invincibility frames after being hit
CRUMBLE_TICKS  = 18    # ticks before a crumble tile disappears after stepped on
LOOKAHEAD      = 25    # cols to generate ahead of right edge
MIN_GAP        = 12    # minimum cols between obstacles (at base speed)

# ---------------------------------------------------------------------------
# Color pair IDs
# ---------------------------------------------------------------------------
C_CABINET  = 1   # TV plastic body — dim white
C_SCREEN   = 2   # screen glass border — cyan
C_SKY      = 3   # star field background — dim
C_GROUND   = 4   # ground tiles — green bold
C_CRUMBLE  = 5   # crumbling tiles — yellow
C_PLAYER   = 6   # player sprite — magenta bold
C_OBSTACLE = 7   # obstacles — red bold
C_COLLECT  = 8   # collectibles — yellow bold
C_HUD      = 9   # HUD bar — white on blue
C_FOOTER   = 10  # footer bar — dim
C_INVULN   = 11  # player while invincible — cyan bold
C_NETCLOG  = 12  # network clog obstacle — cyan


# ---------------------------------------------------------------------------
# SLURM statistics — fetched once at startup
# ---------------------------------------------------------------------------

@dataclass
class SlurmStats:
    """Real-time cluster statistics used to set game difficulty."""
    cpu_alloc:      int   = 0
    cpu_total:      int   = 1      # avoid div-by-zero
    cpu_pct:        float = 0.5    # default to medium difficulty
    jobs_running:   int   = 0
    jobs_pending:   int   = 100
    nodes_draining: int   = 0
    nodes_down:     int   = 0
    nodes_total:    int   = 1
    user_jobs:      int   = 0


def fetch_slurm_stats() -> SlurmStats:
    """
    Query sinfo and squeue for current cluster state.
    Returns safe defaults if any command fails or times out.
    """
    s = SlurmStats()
    try:
        # CPU utilization: sinfo -o "%C" returns "alloc/idle/other/total"
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

        # Node states: count draining and down nodes
        r2 = subprocess.run(
            ['sinfo', '--noheader', '-o', '%t'],
            capture_output=True, text=True, timeout=3
        )
        if r2.returncode == 0:
            lines = r2.stdout.strip().splitlines()
            s.nodes_total    = len(lines)
            s.nodes_draining = sum(1 for l in lines if l.strip() in ('drng', 'drain'))
            s.nodes_down     = sum(1 for l in lines if l.strip() in ('down*', 'down', 'inval'))

        # Job queue: count RUNNING and PENDING
        r3 = subprocess.run(
            ['squeue', '--noheader', '-o', '%T'],
            capture_output=True, text=True, timeout=3
        )
        if r3.returncode == 0:
            states = [l.strip() for l in r3.stdout.strip().splitlines()]
            s.jobs_running = states.count('RUNNING')
            s.jobs_pending = states.count('PENDING')

        # User's own jobs (cosmetic — gives player a hat if they have jobs running)
        user = os.environ.get('USER', '')
        if user:
            r4 = subprocess.run(
                ['squeue', '--noheader', '-u', user, '-o', '%T'],
                capture_output=True, text=True, timeout=3
            )
            if r4.returncode == 0:
                s.user_jobs = len(r4.stdout.strip().splitlines())

    except Exception:
        pass

    return s


def difficulty_label(pct: float) -> str:
    """Human-readable difficulty based on cluster load."""
    if pct < 0.25:
        return 'QUIET  '
    elif pct < 0.50:
        return 'MEDIUM '
    elif pct < 0.75:
        return 'HEAVY  '
    else:
        return 'EXTREME'


# ---------------------------------------------------------------------------
# World objects
# ---------------------------------------------------------------------------

@dataclass
class GroundTile:
    """
    One column of ground at a given world_x position.
    Normal tiles are solid ▓. Crumble tiles (░) collapse after the player
    steps on them. Pit=True means no tile — gap in the ground.
    """
    world_x:       float
    crumble:       bool  = False
    crumble_timer: int   = -1     # ticks since stepped on; -1 = not triggered
    pit:           bool  = False

    @property
    def char(self) -> str:
        if self.pit:
            return ' '
        if not self.crumble:
            return '▓'
        if self.crumble_timer < 0:
            return '░'        # intact crumble tile
        if self.crumble_timer < 6:
            return '·'        # nearly gone
        return '░'

    @property
    def solid(self) -> bool:
        """Whether the player can stand on this tile."""
        if self.pit:
            return False
        if self.crumble and self.crumble_timer >= CRUMBLE_TICKS:
            return False       # fully collapsed
        return True


@dataclass
class Obstacle:
    """
    An obstacle block the player must avoid.
    kind:     'err' | 'mem' | 'oom' | 'net'
    overhead: True means it's above the player and must be ducked under.
    """
    world_x:  float
    kind:     str
    width:    int   = 1
    height:   int   = 2    # world units tall (for collision)
    overhead: bool  = False  # True for ▄▄▄ — duck to avoid

    def chars(self) -> List[str]:
        """Return list of strings to draw, bottom row first."""
        if self.kind == 'err':
            return ['█' * (self.width + 1), '▐' + '▌' * self.width]
        elif self.kind == 'mem':
            return ['▄' * (self.width + 2)]
        elif self.kind == 'oom':
            return ['█' * (self.width + 2), '█' * (self.width + 2), '▄' * (self.width + 2)]
        elif self.kind == 'net':
            return ['≋' * (self.width + 2)]
        return ['??']


@dataclass
class Collectible:
    """
    A floating pickup the player can grab by running through it.
    kind:    'cpu' (+10 pts) | 'node' (+50 pts + invincibility)
    world_y: height above ground in world units
    """
    world_x: float
    kind:    str
    world_y: float = 2.0

    def char(self) -> str:
        return '©' if self.kind == 'cpu' else '[N]'


# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------

class LevelGenerator:
    """
    Procedurally generates ground tiles and obstacles ahead of the player.
    Difficulty parameters come from SlurmStats.

    Key rules to keep the game fair:
    - Always a guaranteed jumpable gap between obstacles
    - Overhead ▄▄▄ obstacles never appear immediately after a pit or crumble tile
    - No two tall obstacles back-to-back (would be impossible)
    """

    def __init__(self, stats: SlurmStats, screen_w: int):
        self.rng          = random.Random()
        self.cursor_x     = float(screen_w)    # next world_x to generate at

        # How many obstacle-seeding events remain from cluster state
        self.pits_budget     = min(stats.nodes_down, 6)
        self.crumbles_budget = min(stats.nodes_draining, 12)

        # Obstacle frequency: 0.0 = rare, 1.0 = frequent
        self.obs_rate = min(0.9, 0.2 + (stats.jobs_pending / 400.0) * 0.7)

        # Minimum gap in world units between obstacle groups
        self.min_gap  = max(8, int(MIN_GAP * (1.0 - stats.cpu_pct * 0.4)))

        self.last_obs_x  = 0.0    # world_x of the last obstacle placed
        self.last_kind   = ''     # kind of the last obstacle (avoid repeats)

    def generate_ahead(self,
                       camera_x: float,
                       screen_w: int,
                       tiles: List[GroundTile],
                       obstacles: List[Obstacle],
                       collectibles: List[Collectible]) -> None:
        """
        Called each tick. Fills the world up to camera_x + screen_w + LOOKAHEAD.
        Appends new objects to the provided lists.
        """
        right_edge = camera_x + screen_w + LOOKAHEAD

        while self.cursor_x < right_edge:
            x = self.cursor_x

            # --- Ground tile ---
            tile = GroundTile(world_x=x)

            # Maybe make it a pit (gap)
            if (self.pits_budget > 0
                    and x > camera_x + screen_w + 4   # not too close
                    and x - self.last_obs_x > self.min_gap + 4
                    and self.rng.random() < 0.04):
                tile.pit = True
                self.pits_budget -= 1
                # Pits are 3–5 tiles wide — generate the rest now
                pit_w = self.rng.randint(2, 4)
                tiles.append(tile)
                for i in range(1, pit_w):
                    tiles.append(GroundTile(world_x=x + i, pit=True))
                self.cursor_x += pit_w
                self.last_obs_x = x
                continue

            # Maybe make it a crumble tile
            elif (self.crumbles_budget > 0
                    and self.rng.random() < 0.06
                    and not (tiles and tiles[-1].pit)):
                tile.crumble = True
                self.crumbles_budget -= 1

            tiles.append(tile)

            # --- Obstacle (only on solid ground, with minimum gap enforced) ---
            gap_ok = (x - self.last_obs_x) >= self.min_gap
            if gap_ok and not tile.pit and self.rng.random() < self.obs_rate / 8:
                obs = self._pick_obstacle(x)
                if obs:
                    obstacles.append(obs)
                    self.last_obs_x = x
                    self.last_kind  = obs.kind

                    # Maybe add a collectible just before this obstacle
                    if self.rng.random() < 0.4:
                        cx   = x - self.rng.randint(3, 6)
                        kind = 'node' if self.rng.random() < 0.1 else 'cpu'
                        collectibles.append(Collectible(world_x=cx, kind=kind))

            self.cursor_x += 1.0

    def _pick_obstacle(self, x: float) -> Optional[Obstacle]:
        """Choose and return an obstacle type, or None."""
        choices = ['err', 'mem', 'oom', 'net']
        if self.last_kind in choices:
            choices = [c for c in choices if c != self.last_kind]

        kind = self.rng.choice(choices)

        if kind == 'err':
            w = self.rng.randint(1, 3)
            return Obstacle(world_x=x, kind='err', width=w, height=2)
        elif kind == 'mem':
            w = self.rng.randint(2, 5)
            return Obstacle(world_x=x, kind='mem', width=w, height=1, overhead=True)
        elif kind == 'oom':
            return Obstacle(world_x=x, kind='oom', width=3, height=3)
        elif kind == 'net':
            w = self.rng.randint(2, 4)
            return Obstacle(world_x=x, kind='net', width=w, height=1)
        return None


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    world_y:   float = 0.0    # 0 = standing on ground
    vy:        float = 0.0    # vertical velocity (positive = up)
    airborne:  bool  = False
    ducking:   bool  = False
    lives:     int   = LIVES_START
    inv_ticks: int   = 0      # invincibility frames remaining
    score:     int   = 0
    distance:  float = 0.0    # total world units scrolled
    anim_f:    int   = 0      # walk animation frame (0 or 1)
    speed_mul: float = 1.0    # movement speed multiplier (0.6 on NetClog)
    has_job:   bool  = False  # cosmetic: user has running jobs (draws hat)

    def jump(self):
        if not self.airborne:
            self.vy       = JUMP_V
            self.airborne = True

    def get_hit(self):
        """Take a hit. Sets invincibility and loses a life."""
        if self.inv_ticks > 0:
            return   # already invincible
        self.lives    -= 1
        self.inv_ticks = INVULN_TICKS
        self.world_y   = 0.0
        self.vy        = 0.0
        self.airborne  = False
        self.ducking   = False

    def sprite(self) -> List[str]:
        """
        Returns list of strings ordered bottom-row FIRST.
        _draw_player iterates with i=0 at ground_row, i=1 one row up, etc.
        So index 0 = feet/legs, index 1 = body, index 2 = head.
        """
        if self.ducking:
            return ['[>_>]']
        if self.airborne:
            return [' /_\\ ', '-o-', ' O ']
        # Walking — two alternating leg frames
        legs = ' /|  ' if self.anim_f == 0 else '  |\\ '
        hat  = '\\o/' if not self.has_job else '[o]'
        return [legs, ' -|- ', f' {hat} ']

    @property
    def height(self) -> int:
        """Collision height in world units."""
        return 1 if self.ducking else 3

    @property
    def width(self) -> int:
        return 5


# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class SlurmRunner:
    """
    Fullscreen SLURM-themed endless runner game.
    Designed to be exec()'d and called as SlurmRunner(stdscr, stats).run().
    """

    CULL_LEFT = -10

    def __init__(self, stdscr, stats: SlurmStats):
        self.stdscr = stdscr
        self.stats  = stats
        self.h, self.w = stdscr.getmaxyx()

        self._compute_layout()

        self.camera_x     = 0.0
        self.scroll_speed = BASE_SPEED + stats.cpu_pct * (MAX_SPEED - BASE_SPEED)
        self.tick_n       = 0
        self.done         = False
        self.player       = Player(has_job=(stats.user_jobs > 0))

        self.tiles:        List[GroundTile]  = []
        self.obstacles:    List[Obstacle]    = []
        self.collectibles: List[Collectible] = []

        self.gen = LevelGenerator(stats, self.sw)
        self.gen.generate_ahead(
            self.camera_x, self.sw,
            self.tiles, self.obstacles, self.collectibles
        )

        self.stars = [
            (self.rng.randint(0, self.sw - 1),
             self.rng.randint(0, max(1, self.sky_rows - 1)))
            for _ in range(max(1, self.sw // 5))
        ]
        self.star_scroll = 0.0

    @property
    def rng(self) -> random.Random:
        if not hasattr(self, '_rng'):
            self._rng = random.Random()
        return self._rng

    def _compute_layout(self):
        """
        Calculate the geometry of the TV cabinet and the play field inside it.

        TV cabinet structure (for 80×24 terminal):
          rows 0–1   : antenna rows
          row  2     : cabinet top (╭──╮)
          row  3     : label row (SLURM-TV)
          row  4     : separator (├──┬──┤)
          rows 5–h-4 : play field (screen left | controls right)
          row  h-3   : bottom separator
          row  h-2   : speaker grille
          row  h-1   : cabinet bottom (╰──╯)
        """
        h, w = self.h, self.w

        self.ctrl_w    = 13
        self.divider   = w - self.ctrl_w - 1

        self.cab_top   = 2
        self.label_row = 3
        self.sep_row   = 4
        self.play_top  = 5
        self.play_bot  = h - 4
        self.grille_row = h - 3
        self.cab_bot   = h - 2

        self.sx0 = 2
        self.sx1 = self.divider - 2
        self.sy0 = self.play_top + 1
        self.sy1 = self.play_bot - 1

        self.sw  = self.sx1 - self.sx0 + 1
        self.sh  = self.sy1 - self.sy0 + 1

        # ground_row: player/obstacles stand here; tiles drawn at ground_row+1; sy1 is footer
        self.ground_row = self.sy1 - 2

        self.sky_rows  = max(1, self.sh // 3)
        self.hud_row   = self.sy0
        self.play_sy0  = self.sy0 + 1

    def _world_to_sx(self, wx: float) -> int:
        return int(wx - self.camera_x)

    def _world_to_screen_col(self, wx: float) -> int:
        return self.sx0 + self._world_to_sx(wx)

    def _world_to_screen_row(self, wy: float) -> int:
        """
        world_y=0 → ground_row (player feet at ground level; tiles at ground_row+1).
        world_y=1 → ground_row - 1, etc.
        """
        return self.ground_row - int(wy)

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def _tick(self, duck_held: bool):
        """Advance physics, scroll world, handle all collisions."""
        self.tick_n += 1

        self.player.ducking = duck_held and not self.player.airborne

        self.scroll_speed = min(MAX_SPEED, self.scroll_speed + SPEED_CREEP)

        scroll = self.scroll_speed * self.player.speed_mul
        self.camera_x        += scroll
        self.player.distance += scroll

        # Vertical physics
        if self.player.airborne or self.player.world_y > 0:
            self.player.vy      -= GRAVITY
            self.player.world_y += self.player.vy

        # Ground collision
        player_gx = int(self.camera_x + PLAYER_COL)
        ground_under = [t for t in self.tiles if int(t.world_x) == player_gx and t.solid]

        if ground_under or (not self.player.airborne and self.player.world_y <= 0):
            if self.player.world_y <= 0:
                self.player.world_y  = 0.0
                self.player.vy       = 0.0
                self.player.airborne = False
                for t in [t for t in self.tiles if int(t.world_x) == player_gx]:
                    if t.crumble and t.crumble_timer < 0:
                        t.crumble_timer = 0

        # Pit death
        tile_here = next((t for t in self.tiles if int(t.world_x) == player_gx), None)
        if tile_here is None or tile_here.pit:
            if self.player.world_y <= -2:
                self.player.get_hit()
                if self.player.lives <= 0:
                    self.done = True
                    return

        # Crumble timers
        for t in self.tiles:
            if t.crumble and t.crumble_timer >= 0:
                t.crumble_timer += 1
                if t.crumble_timer >= CRUMBLE_TICKS:
                    t.pit    = True
                    t.crumble = False

        if self.player.inv_ticks > 0:
            self.player.inv_ticks -= 1

        # NetClog speed penalty
        self.player.speed_mul = 1.0
        for obs in self.obstacles:
            sx = self._world_to_sx(obs.world_x)
            if obs.kind == 'net' and -1 <= sx - PLAYER_COL <= obs.width + 1:
                if self.player.world_y <= 0.5 and not self.player.airborne:
                    self.player.speed_mul = 0.65

        # Walk animation
        if self.tick_n % 4 == 0 and not self.player.airborne:
            self.player.anim_f = 1 - self.player.anim_f

        self.player.score += 1

        # Obstacle collision
        if self.player.inv_ticks == 0:
            for obs in self.obstacles:
                if self._check_obs_collision(obs):
                    self.player.get_hit()
                    if self.player.lives <= 0:
                        self.done = True
                        return
                    break

        # Collectible collection
        for col in list(self.collectibles):
            cx = self._world_to_sx(col.world_x)
            if abs(cx - PLAYER_COL) <= 3 and abs(col.world_y - self.player.world_y) <= 2:
                if col.kind == 'cpu':
                    self.player.score += 10
                else:
                    self.player.score    += 50
                    self.player.inv_ticks = INVULN_TICKS
                self.collectibles.remove(col)

        self.gen.generate_ahead(
            self.camera_x, self.sw,
            self.tiles, self.obstacles, self.collectibles
        )

        cutoff = self.camera_x + self.CULL_LEFT
        self.tiles        = [t for t in self.tiles        if t.world_x >= cutoff]
        self.obstacles    = [o for o in self.obstacles    if o.world_x + o.width >= cutoff]
        self.collectibles = [c for c in self.collectibles if c.world_x >= cutoff]

        self.star_scroll += scroll * 0.5

    def _check_obs_collision(self, obs: Obstacle) -> bool:
        """AABB collision between player and obstacle."""
        sx = self._world_to_sx(obs.world_x)
        px = PLAYER_COL

        if not (sx - self.player.width // 2 < px + self.player.width // 2
                and sx + obs.width > px - self.player.width // 2):
            return False

        if obs.overhead:
            if self.player.ducking:
                return False
            if self.player.world_y > 3:
                return False
            return True
        else:
            obs_top    = obs.height
            player_top = self.player.world_y + self.player.height
            if player_top < 0.2:
                return False
            return player_top > 0 and self.player.world_y < obs_top

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _init_colors(self):
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(C_CABINET,  curses.COLOR_WHITE,   -1)
        curses.init_pair(C_SCREEN,   curses.COLOR_CYAN,    -1)
        curses.init_pair(C_SKY,      curses.COLOR_WHITE,   -1)
        curses.init_pair(C_GROUND,   curses.COLOR_GREEN,   -1)
        curses.init_pair(C_CRUMBLE,  curses.COLOR_YELLOW,  -1)
        curses.init_pair(C_PLAYER,   curses.COLOR_MAGENTA, -1)
        curses.init_pair(C_OBSTACLE, curses.COLOR_RED,     -1)
        curses.init_pair(C_COLLECT,  curses.COLOR_YELLOW,  -1)
        curses.init_pair(C_HUD,      curses.COLOR_WHITE,   curses.COLOR_BLUE)
        curses.init_pair(C_FOOTER,   curses.COLOR_WHITE,   -1)
        curses.init_pair(C_INVULN,   curses.COLOR_CYAN,    -1)
        curses.init_pair(C_NETCLOG,  curses.COLOR_CYAN,    -1)

    def _attr(self, pair_id: int, bold: bool = False, dim: bool = False) -> int:
        attr = curses.color_pair(pair_id) if curses.has_colors() else curses.A_NORMAL
        if bold:
            attr |= curses.A_BOLD
        if dim:
            attr |= curses.A_DIM
        return attr

    def _safe_addstr(self, row: int, col: int, s: str, attr: int = 0) -> None:
        h, w = self.h, self.w
        if row < 0 or row >= h - 1:
            return
        if col >= w - 1:
            return
        if col < 0:
            s = s[-col:]
            col = 0
        s = s[:max(0, w - 1 - col)]
        if not s:
            return
        try:
            self.stdscr.addstr(row, col, s, attr)
        except curses.error:
            pass

    def _draw_tv_cabinet(self):
        """
        Draw the CRT TV cabinet that surrounds the game.

        Layout:
          Row 0-1  : rabbit-ear antennas
          Row 2    : cabinet top  ╭──────────────────────────────────────────╮
          Row 3    : label row    │              S L U R M - T V             │
          Row 4    : separator    ├────────────────────────┬─────────────────┤
          Row 5..  : play field   │  [screen]              │ CH+  VOL  PWR   │
          Row h-3  : grille sep   ├────────────────────────┴─────────────────┤
          Row h-2  : grille       │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
          Row h-1  : cabinet bot  ╰──────────────────────────────────────────╯
        """
        h, w = self.h, self.w
        cab_bold = self._attr(C_CABINET, bold=True)
        dim      = self._attr(C_CABINET, dim=True)
        scr      = self._attr(C_SCREEN, bold=True)

        # Antennas — left leans left, right leans right
        ant_l = w // 4
        ant_r = w - w // 4
        self._safe_addstr(0, ant_l - 2, '/\\', dim)
        self._safe_addstr(1, ant_l - 3, '/',   dim)
        self._safe_addstr(0, ant_r,      '/\\', dim)
        self._safe_addstr(1, ant_r + 2,  '\\',  dim)

        # Cabinet top
        self._safe_addstr(self.cab_top, 0,
                          ('╭' + '─' * (w - 2) + '╮')[:w - 1], cab_bold)

        # Label row
        label = 'S  L  U  R  M  -  T  V'
        self._safe_addstr(self.label_row, 0,
                          ('│' + label.center(w - 2) + '│')[:w - 1], cab_bold)

        # Separator with divider T-junction
        dv = self.divider
        sep = '├' + '─' * (dv - 1) + '┬' + '─' * (w - dv - 2) + '┤'
        self._safe_addstr(self.sep_row, 0, sep[:w - 1], cab_bold)

        # Side borders for every play field row
        for r in range(self.play_top, self.play_bot + 1):
            self._safe_addstr(r, 0,   '│', cab_bold)
            self._safe_addstr(r, dv,  '│', cab_bold)
            self._safe_addstr(r, w - 1, '│', cab_bold)

        # Screen glass (rounded corners)
        gl, gr = 1, dv - 1
        gt, gb = self.play_top, self.play_bot
        self._safe_addstr(gt, gl, '╭' + '─' * (gr - gl - 1) + '╮', scr)
        self._safe_addstr(gb, gl, '╰' + '─' * (gr - gl - 1) + '╯', scr)
        for r in range(gt + 1, gb):
            self._safe_addstr(r, gl, '│', scr)
            self._safe_addstr(r, gr, '│', scr)

        # Scanline background inside screen interior
        for r in range(self.sy0, self.sy1 + 1):
            fill = ' ' * (self.sx1 - self.sx0 + 1)
            self._safe_addstr(r, self.sx0, fill,
                              dim if r % 2 == 0 else curses.A_NORMAL)

        # Control panel
        cx = dv + 2
        ph = self.play_bot - self.play_top
        ch_row  = self.play_top + max(1, ph // 5)
        vol_row = self.play_top + max(2, ph * 2 // 5)
        pwr_row = self.play_top + max(3, ph * 3 // 5)
        spd_row = self.play_top + max(4, ph * 4 // 5)

        self._safe_addstr(ch_row - 1,  cx, 'CH+',   cab_bold)
        self._safe_addstr(ch_row,      cx, '(   )',  cab_bold)
        self._safe_addstr(vol_row - 1, cx, 'VOL',   cab_bold)
        self._safe_addstr(vol_row,     cx, '(   )',  cab_bold)
        self._safe_addstr(pwr_row - 1, cx, 'PWR',   cab_bold)
        self._safe_addstr(pwr_row,     cx, '[ ■ ]', cab_bold)
        self._safe_addstr(spd_row,     cx, 'SPD',   dim)
        self._safe_addstr(spd_row + 1, cx, f'{self.scroll_speed:.1f}x', cab_bold)

        # Cluster stats panel
        cw = w - self.divider - 3   # available width in ctrl column
        r = spd_row + 3
        nd_total = max(1, self.stats.nodes_total)
        nd_up    = nd_total - self.stats.nodes_down - self.stats.nodes_draining
        self._safe_addstr(r,     cx, 'NODES', dim)
        self._safe_addstr(r + 1, cx, f'{nd_up}/{nd_total}'[:cw], cab_bold)
        self._safe_addstr(r + 3, cx, 'JOBS',  dim)
        self._safe_addstr(r + 4, cx, f'{self.stats.jobs_running}R'[:cw], cab_bold)
        self._safe_addstr(r + 5, cx, f'{self.stats.jobs_pending}P'[:cw], dim)
        pw_bar = 8
        filled = int(self.stats.cpu_pct * pw_bar)
        bar    = '█' * filled + '░' * (pw_bar - filled)
        self._safe_addstr(r + 7, cx, 'LOAD',  dim)
        self._safe_addstr(r + 8, cx, bar[:cw], self._attr(C_HUD, bold=True))

        # Bottom grille separator
        bot_sep = '├' + '─' * (dv - 1) + '┴' + '─' * (w - dv - 2) + '┤'
        self._safe_addstr(self.grille_row, 0, bot_sep[:w - 1], cab_bold)

        # Speaker grille
        self._safe_addstr(self.grille_row + 1, 1,
                          ('░' * (w - 2))[:w - 2], dim)

        # Cabinet bottom
        self._safe_addstr(self.cab_bot, 0,
                          ('╰' + '─' * (w - 2) + '╯')[:w - 1], cab_bold)

    def _draw_hud(self):
        hearts  = '♥' * self.player.lives + '♡' * (LIVES_START - self.player.lives)
        pw      = 10
        filled  = int(self.stats.cpu_pct * pw)
        bar     = '█' * filled + '░' * (pw - filled)
        hud = (f' {hearts}  CLUSTER:{bar}{int(self.stats.cpu_pct*100):2d}%'
               f'  PEND:{self.stats.jobs_pending}'
               f'  SCORE:{self.player.score:06d} ')
        self._safe_addstr(self.hud_row, self.sx0,
                          hud[:self.sw].ljust(self.sw),
                          self._attr(C_HUD, bold=True))

    def _draw_stars(self):
        dim = self._attr(C_SKY, dim=True)
        for (sc, sr) in self.stars:
            col = self.sx0 + int((sc - self.star_scroll * 0.5) % self.sw)
            row = self.play_sy0 + sr
            if self.play_sy0 <= row < self.play_sy0 + self.sky_rows:
                self._safe_addstr(row, col, '.', dim)

    def _draw_ground(self):
        for tile in self.tiles:
            col = self._world_to_screen_col(tile.world_x)
            if self.sx0 <= col <= self.sx1 and not tile.pit:
                attr = (self._attr(C_CRUMBLE, bold=True) if tile.crumble
                        else self._attr(C_GROUND, bold=True))
                self._safe_addstr(self.ground_row + 1, col, tile.char, attr)

    def _draw_obstacles(self):
        for obs in self.obstacles:
            col = self._world_to_screen_col(obs.world_x)
            if col > self.sx1 or col + obs.width < self.sx0:
                continue

            lines = obs.chars()

            if obs.overhead:
                base_row = self._world_to_screen_row(2)
                self._safe_addstr(base_row, col, lines[0],
                                  self._attr(C_OBSTACLE, bold=True))
            else:
                attr     = (self._attr(C_NETCLOG, bold=True) if obs.kind == 'net'
                            else self._attr(C_OBSTACLE, bold=True))
                base_row = self.ground_row
                for i, line in enumerate(lines):
                    row = base_row - i
                    if self.play_sy0 <= row <= self.ground_row:
                        self._safe_addstr(row, col, line, attr)

    def _draw_collectibles(self):
        attr = self._attr(C_COLLECT, bold=True)
        for col_obj in self.collectibles:
            col = self._world_to_screen_col(col_obj.world_x)
            row = self._world_to_screen_row(col_obj.world_y)
            if self.sx0 <= col <= self.sx1 and self.play_sy0 <= row <= self.ground_row:
                self._safe_addstr(row, col, col_obj.char(), attr)

    def _draw_player(self):
        if self.player.inv_ticks > 0 and self.tick_n % 6 < 3:
            return   # blink during invincibility

        attr   = (self._attr(C_INVULN, bold=True) if self.player.inv_ticks > 0
                  else self._attr(C_PLAYER, bold=True))
        sprite = self.player.sprite()
        base_row = self._world_to_screen_row(self.player.world_y)
        col = self.sx0 + PLAYER_COL - len(sprite[0]) // 2

        for i, line in enumerate(sprite):
            row = base_row - i
            if self.play_sy0 <= row <= self.ground_row:
                self._safe_addstr(row, col, line, attr)

    def _draw_footer(self):
        dist_km = self.player.distance / 100.0
        footer  = (f'  SPEED:{self.scroll_speed:.2f}x  '
                   f'DISTANCE:{dist_km:.1f}km  '
                   f'LOAD:{difficulty_label(self.stats.cpu_pct)}  '
                   f'NODES:{self.stats.nodes_total}')
        self._safe_addstr(self.sy1, self.sx0,
                          footer[:self.sx1 - self.sx0 + 1],
                          self._attr(C_FOOTER, dim=True))

    def _draw(self):
        self.stdscr.erase()
        self._draw_tv_cabinet()
        self._draw_stars()
        self._draw_ground()
        self._draw_obstacles()
        self._draw_collectibles()
        self._draw_player()
        self._draw_hud()
        self._draw_footer()
        self.stdscr.refresh()

    # ------------------------------------------------------------------
    # Game over screen
    # ------------------------------------------------------------------

    def _game_over(self) -> bool:
        """
        Display game over screen inside the TV.
        Returns True if player wants to play again, False to quit.
        """
        self.stdscr.erase()
        self._draw_tv_cabinet()

        lines = [
            r"  ____    _    __  __ _____     _____   _     _______ ____  ",
            r" / ___|  / \  |  \/  | ____|   / _ \ \ / /   | ____||  _ \ ",
            r"| |  _  / _ \ | |\/| |  _|    | | | \ V /    |  _|  | |_) |",
            r"| |_| |/ ___ \| |  | | |___   | |_| || |     | |___ |  _ < ",
            r" \____/_/   \_\_|  |_|_____|   \___/ |_|     |_____||_| \_\\",
            "",
            f"  SCORE: {self.player.score:06d}",
            f"  DISTANCE: {self.player.distance / 100:.1f} km",
            f"  CLUSTER LOAD: {int(self.stats.cpu_pct * 100)}%  ({difficulty_label(self.stats.cpu_pct).strip()})",
            "",
            "  [R] Play again    [Q] Quit",
        ]

        start_row = self.sy0 + max(0, (self.sh - len(lines)) // 2)
        attr = self._attr(C_OBSTACLE, bold=True)

        for i, line in enumerate(lines):
            row = start_row + i
            if row > self.sy1:
                break
            col = self.sx0 + max(0, (self.sw - len(line)) // 2)
            a = attr if i < 5 else (self._attr(C_FOOTER, dim=True) if '[R]' in line
                                    else self._attr(C_HUD))
            self._safe_addstr(row, col, line[:self.sx1 - col + 1], a)

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

    def _reset(self):
        """Reset all game state for a fresh run (keeps stats/layout)."""
        self.camera_x     = 0.0
        self.scroll_speed = BASE_SPEED + self.stats.cpu_pct * (MAX_SPEED - BASE_SPEED)
        self.tick_n       = 0
        self.done         = False
        self.player       = Player(has_job=(self.stats.user_jobs > 0))
        self.tiles        = []
        self.obstacles    = []
        self.collectibles = []
        self.gen          = LevelGenerator(self.stats, self.sw)
        self.star_scroll  = 0.0
        self.gen.generate_ahead(
            self.camera_x, self.sw,
            self.tiles, self.obstacles, self.collectibles
        )

    def run(self):
        """
        Main game loop. Uses halfdelay(1) = 100ms timeout on getch().
        Loops back to a fresh game if the player chooses to replay.
        """
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
                    self.player.jump()
                    duck_held = False
                elif key in (curses.KEY_DOWN, ord('s')):
                    duck_held = True

                self._tick(duck_held)

            play_again = self._game_over()
            if not play_again:
                return
            self._reset()
