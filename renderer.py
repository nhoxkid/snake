# renderer.py - Game renderer + honest neural network monitor
#
# What's real here:
# - Node activations: actual values from the forward pass
# - Input saliency: real gradient |dQ/d(input)| per Simonyan et al. 2014
# - LSTM cell state: the actual memory vector (not hidden state)
# - Connections Dense1→Output: actual weight matrix values
# - Connections LSTM→Dense1: actual weight matrix values (sampled rows)
# - Q-values + deltas: real Q-values, real step-to-step difference
# - Loss curve: real training loss over time
#
# What's approximated:
# - Input→LSTM connections: shown via saliency (gradient) not raw LSTM kernel,
#   because LSTM has 4 internal gates that can't be meaningfully drawn as lines.
# - Hidden layers show 8 of 64 neurons (fixed indices 0-7 for consistency).
from __future__ import annotations
import pygame
import numpy as np
from config import GAME, TRAIN

# ── Colors ───────────────────────────────────────────────────────────
_BG        = (18, 18, 24)
_GRID      = (25, 25, 30)
_WHITE     = (220, 220, 220)
_DIM       = (90, 90, 100)
_RED       = (255, 60, 60)
_GREEN     = (60, 220, 80)
_CYAN      = (0, 200, 220)
_YELLOW    = (255, 220, 40)
_ORANGE    = (255, 140, 40)
_BLUE      = (50, 100, 255)

_INPUT_LABELS = [
    'dir_U', 'dir_D', 'dir_L', 'dir_R',
    'fd_dx', 'fd_dy',
    'w_up', 'w_dn', 'w_lt', 'w_rt',
    'dng_S', 'dng_L', 'dng_R',
    'body',
]
_ACTION_LABELS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Fixed neuron indices for hidden layers (consistent across frames)
_VIS_HIDDEN = 8
_HIDDEN_IDX = list(range(_VIS_HIDDEN))


def _clamp(v, lo=0, hi=255):
    return int(max(lo, min(hi, v)))


def _val_to_color(v: float) -> tuple[int, int, int]:
    """Diverging colormap: blue (neg) → dark (zero) → green (pos)."""
    if v >= 0:
        t = min(v, 1.0)
        return (0, _clamp(200 * t), _clamp(60 * t))
    else:
        t = min(-v, 1.0)
        return (_clamp(40 * t), _clamp(60 * t), _clamp(220 * t))


def _weight_line_color(w: float) -> tuple[int, int, int]:
    """Red (neg weight) → transparent (zero) → cyan (pos weight)."""
    if w >= 0:
        t = min(w, 1.0)
        return (_clamp(20 + 20 * (1 - t)), _clamp(60 + 140 * t), _clamp(80 + 175 * t))
    else:
        t = min(-w, 1.0)
        return (_clamp(60 + 195 * t), _clamp(30 * (1 - t)), _clamp(30 * (1 - t)))


# ── Game Renderer ────────────────────────────────────────────────────

class GameRenderer:
    """Draws snake game from a thread-safe GameSnapshot."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        self._font: pygame.font.Font | None = None

    @property
    def font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.SysFont('consolas', 14)
        return self._font

    def draw(self, gs) -> None:
        """Draw from a GameSnapshot (thread-safe, no live object access)."""
        self.surface.fill(_BG)
        cell = GAME.cell

        for x in range(0, GAME.width, cell):
            pygame.draw.line(self.surface, _GRID, (x, 0), (x, GAME.height))
        for y in range(0, GAME.height, cell):
            pygame.draw.line(self.surface, _GRID, (0, y), (GAME.width, y))

        n = len(gs.body)
        for i, seg in enumerate(gs.body):
            t = 1.0 - (i / max(n, 1))
            g = int(80 + 175 * t)
            pygame.draw.rect(self.surface, (0, g, _clamp(40 * t)),
                             (seg[0] + 1, seg[1] + 1, cell - 2, cell - 2))

        fx, fy = gs.food_pos
        pygame.draw.rect(self.surface, _RED, (fx + 1, fy + 1, cell - 2, cell - 2))

        hud = (f"Ep {gs.episode}  Step {gs.step}  "
               f"Score {gs.score}  R{gs.reward:+.2f}  \u03b5{gs.epsilon:.3f}")
        self.surface.blit(self.font.render(hud, True, _WHITE), (6, 3))


# ── Network Visualizer ───────────────────────────────────────────────

class NetworkVisualizer:
    """Honest neural network monitor.

    Every value shown comes from the actual forward pass or real weight matrices.
    Saliency uses vanilla gradient (Simonyan 2014). LSTM cell state is the real
    memory vector. Connections use actual Dense layer kernels.
    """

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        self.w = surface.get_width()
        self.h = surface.get_height()
        self._fonts: dict[int, pygame.font.Font] = {}
        self._loss_hist: list[float] = []
        self._td_hist: list[float] = []

    def _f(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont('consolas', size)
        return self._fonts[size]

    def draw(self, snap) -> None:
        self.surface.fill(_BG)
        if snap is None:
            return

        if snap.loss > 0:
            self._loss_hist.append(snap.loss)
        if snap.td_error > 0:
            self._td_hist.append(snap.td_error)

        # Layout: network graph (top 50%), LSTM memory (8%), Q-bars (20%), info (22%)
        y_net = 0
        h_net = int(self.h * 0.48)
        y_mem = h_net + 2
        h_mem = int(self.h * 0.10)
        y_qbar = y_mem + h_mem + 2
        h_qbar = int(self.h * 0.20)
        y_info = y_qbar + h_qbar + 2
        h_info = self.h - y_info

        self._draw_network(snap, y_net, h_net)
        self._draw_lstm_memory(snap, y_mem, h_mem)
        self._draw_qbars(snap, y_qbar, h_qbar)
        self._draw_info(snap, y_info, h_info)

    # ── Network Graph ────────────────────────────────────────────

    def _draw_network(self, snap, y0, h):
        mx, my = 52, 16
        n_in = TRAIN.state_size
        n_hid = _VIS_HIDDEN
        n_out = TRAIN.n_actions
        layer_sizes = [n_in, n_hid, n_hid, n_out]
        headers = ['INPUT', 'LSTM[0:8]', 'DENSE[0:8]', 'Q-OUT']

        usable_w = self.w - 2 * mx
        xs = [mx + int(usable_w * i / 3) for i in range(4)]

        def ypos(cx, count):
            usable = h - 2 * my - 14
            top = y0 + my + 14
            if count == 1:
                return [(cx, top + usable // 2)]
            return [(cx, top + int(usable * i / (count - 1))) for i in range(count)]

        layers = [ypos(xs[i], layer_sizes[i]) for i in range(4)]

        # Activations (fixed indices for hidden layers)
        acts = [
            snap.input_vals,
            snap.lstm_hidden[_HIDDEN_IDX],
            snap.dense_act[_HIDDEN_IDX],
            snap.q_values,
        ]

        # ── Connections using REAL weight matrices ──

        # LSTM→Dense1: actual dense1_kernel[0:8, 0:8]
        self._draw_weight_connections(
            layers[1], layers[2], snap.dense1_kernel, _HIDDEN_IDX, _HIDDEN_IDX)

        # Dense1→Output: actual output_kernel[0:8, 0:4]
        self._draw_weight_connections(
            layers[2], layers[3], snap.output_kernel, _HIDDEN_IDX, list(range(n_out)))

        # Input→LSTM: use SALIENCY as connection strength (honest: we can't
        # meaningfully decompose LSTM gate weights into simple lines)
        self._draw_saliency_connections(
            layers[0], layers[1], snap.input_saliency, acts[1])

        # ── Headers ──
        f_hdr = self._f(8)
        for i, label in enumerate(headers):
            txt = f_hdr.render(label, True, _DIM)
            self.surface.blit(txt, (xs[i] - txt.get_width() // 2, y0 + 2))

        # ── Nodes ──
        f_val = self._f(9)
        saliency_max = snap.input_saliency.max() or 1.0

        for li, (layer_pos, act_vals) in enumerate(zip(layers, acts)):
            for ni, (pos, val) in enumerate(zip(layer_pos, act_vals)):
                vf = float(val)
                color = _val_to_color(vf)

                # Base radius; input nodes also scale with saliency
                r = 7
                if li == 0:
                    sal = float(snap.input_saliency[ni]) / saliency_max
                    r = 5 + int(sal * 7)  # 5..12 based on saliency
                    # Saliency glow
                    if sal > 0.3:
                        gr = r + 5
                        gs = pygame.Surface((gr * 2, gr * 2), pygame.SRCALPHA)
                        alpha = _clamp(sal * 100, 0, 100)
                        pygame.draw.circle(gs, (*_YELLOW, alpha), (gr, gr), gr)
                        self.surface.blit(gs, (pos[0] - gr, pos[1] - gr))

                pygame.draw.circle(self.surface, color, pos, r)
                pygame.draw.circle(self.surface, _DIM, pos, r, 1)

                # Value labels
                vs = f'{vf:+.2f}'
                vt = f_val.render(vs, True, _WHITE)

                if li == 0:  # Input: label left, value right
                    lbl = _INPUT_LABELS[ni] if ni < len(_INPUT_LABELS) else f'i{ni}'
                    lt = f_val.render(lbl, True, _DIM)
                    self.surface.blit(lt, (pos[0] - r - lt.get_width() - 2, pos[1] - 5))
                    self.surface.blit(vt, (pos[0] + r + 2, pos[1] - 5))
                elif li == 3:  # Output: action label + Q-value
                    chosen = ni == snap.chosen_action
                    c = _YELLOW if chosen else _DIM
                    lt = f_val.render(_ACTION_LABELS[ni], True, c)
                    self.surface.blit(lt, (pos[0] + r + 3, pos[1] - 10))
                    self.surface.blit(vt, (pos[0] + r + 3, pos[1] + 1))
                    if chosen:
                        pygame.draw.circle(self.surface, _YELLOW, pos, r + 3, 2)
                else:  # Hidden: value below
                    self.surface.blit(vt, (pos[0] - vt.get_width() // 2, pos[1] + r + 1))

    def _draw_weight_connections(self, from_pos, to_pos, kernel, from_idx, to_idx):
        """Draw connections using ACTUAL weight matrix values."""
        w_max = np.abs(kernel[np.ix_(from_idx, to_idx)]).max() or 1.0
        for i, fi in enumerate(from_idx):
            if i >= len(from_pos):
                break
            for j, tj in enumerate(to_idx):
                if j >= len(to_pos):
                    break
                w = float(kernel[fi, tj])
                norm_w = w / w_max
                if abs(norm_w) < 0.05:
                    continue
                color = _weight_line_color(norm_w)
                thick = max(1, int(abs(norm_w) * 3))
                pygame.draw.line(self.surface, color, from_pos[i], to_pos[j], thick)

    def _draw_saliency_connections(self, from_pos, to_pos, saliency, to_act):
        """Input→LSTM: line brightness = saliency of that input feature.
        This is honest: we can't decompose LSTM gates into simple lines,
        so we show gradient-based importance instead."""
        sal_max = saliency.max() or 1.0
        for i, (fp, sal) in enumerate(zip(from_pos, saliency)):
            s = float(sal) / sal_max
            if s < 0.05:
                continue
            for j, tp in enumerate(to_pos):
                alpha = _clamp(s * 180, 30, 180)
                color = (_clamp(s * 255), _clamp(s * 200), _clamp(50 + s * 100))
                thick = max(1, int(s * 2))
                pygame.draw.line(self.surface, color, fp, tp, thick)

    # ── LSTM Cell State (Memory) ─────────────────────────────────

    def _draw_lstm_memory(self, snap, y0, h):
        f = self._f(9)
        margin = 10

        # Header
        self.surface.blit(f.render('LSTM Cell State (memory)', True, _DIM),
                          (margin, y0 + 1))

        # Heatmap of all 64 cell state values
        cell = snap.lstm_cell
        n = len(cell)
        cell_w = max(1, (self.w - 2 * margin) // n)
        bar_h = min(16, h - 26)
        bar_y = y0 + 14

        c_max = max(np.abs(cell).max(), 0.01)
        for i in range(n):
            v = float(cell[i]) / c_max
            color = _val_to_color(v)
            x = margin + i * cell_w
            pygame.draw.rect(self.surface, color, (x, bar_y, cell_w - 1, bar_h))

        # Hidden state below
        hy = bar_y + bar_h + 2
        if hy + 14 < y0 + h:
            self.surface.blit(f.render('LSTM Hidden State (output)', True, _DIM),
                              (margin, hy))
            hidden = snap.lstm_hidden
            h_max = max(np.abs(hidden).max(), 0.01)
            hbar_h = min(14, y0 + h - hy - 14)
            for i in range(n):
                v = float(hidden[i]) / h_max
                color = _val_to_color(v)
                x = margin + i * cell_w
                pygame.draw.rect(self.surface, color, (x, hy + 12, cell_w - 1, hbar_h))

    # ── Q-Value Bars ─────────────────────────────────────────────

    def _draw_qbars(self, snap, y0, h):
        f_title = self._f(11)
        f = self._f(10)
        margin = 10

        mode = 'RANDOM' if snap.was_random else 'GREEDY'
        mc = _ORANGE if snap.was_random else _CYAN
        self.surface.blit(f_title.render(f'Q-Values [{mode}]', True, mc),
                          (margin, y0))

        bar_top = y0 + 16
        n = TRAIN.n_actions
        bar_h = max(10, (h - 20) // n - 4)
        q_abs_max = max(np.abs(snap.q_values).max(), 0.1)

        for i in range(n):
            by = bar_top + i * (bar_h + 4)
            q = float(snap.q_values[i])
            chosen = i == snap.chosen_action

            # Label
            c = _YELLOW if chosen else _DIM
            self.surface.blit(f.render(f'{_ACTION_LABELS[i]}:', True, c), (margin, by))

            # Bar bg
            bx = margin + 42
            bw = self.w - margin * 2 - 150
            pygame.draw.rect(self.surface, (30, 30, 38), (bx, by, bw, bar_h))

            # Bar fill centered at midpoint
            mid = bx + bw // 2
            fill = int((q / q_abs_max) * (bw // 2))
            if fill > 0:
                bc = _GREEN if chosen else (40, 140, 50)
                pygame.draw.rect(self.surface, bc, (mid, by + 1, fill, bar_h - 2))
            elif fill < 0:
                bc = _RED if chosen else (140, 40, 40)
                pygame.draw.rect(self.surface, bc, (mid + fill, by + 1, -fill, bar_h - 2))
            pygame.draw.line(self.surface, _DIM, (mid, by), (mid, by + bar_h), 1)

            # Value
            self.surface.blit(f.render(f'{q:+.4f}', True, _WHITE), (bx + bw + 4, by))

            # Delta
            if snap.prev_q is not None:
                delta = q - float(snap.prev_q[i])
                if abs(delta) > 0.0001:
                    dc = _GREEN if delta > 0 else _RED
                    self.surface.blit(f.render(f'\u0394{delta:+.4f}', True, dc),
                                      (bx + bw + 70, by))

            if chosen:
                pygame.draw.rect(self.surface, _YELLOW,
                                 (bx - 1, by - 1, bw + 2, bar_h + 2), 1)

    # ── Info + Loss Chart ────────────────────────────────────────

    def _draw_info(self, snap, y0, h):
        f = self._f(9)
        margin = 10
        y = y0 + 2

        # Stats row
        parts = []
        if snap.loss > 0:
            parts.append(f'Loss:{snap.loss:.5f}')
        else:
            parts.append('Loss: warmup')
        if snap.td_error > 0:
            parts.append(f'TD:{snap.td_error:.4f}')
        parts.append(f'Act:{_ACTION_LABELS[snap.chosen_action]}')
        parts.append('RND' if snap.was_random else 'GRD')
        self.surface.blit(f.render('  '.join(parts), True, _DIM), (margin, y))
        y += 14

        # Loss curve
        chart_h = h - 20
        if chart_h > 15 and len(self._loss_hist) > 2:
            cw = self.w - 2 * margin
            pygame.draw.rect(self.surface, (22, 22, 28), (margin, y, cw, chart_h))

            self._draw_chart_line(self._loss_hist, margin, y, cw, chart_h, _CYAN, 'Loss')
            if len(self._td_hist) > 2:
                self._draw_chart_line(self._td_hist, margin, y, cw, chart_h, _ORANGE, 'TD err')

    def _draw_chart_line(self, data, x0, y0, w, h, color, label):
        f = self._f(8)
        n = min(len(data), w)
        vals = data[-n:]
        vmax = max(vals) or 1.0
        vmin = min(vals)
        rng = vmax - vmin or 1.0
        pts = []
        for i, v in enumerate(vals):
            px = x0 + int(w * i / max(len(vals) - 1, 1))
            py = y0 + h - 2 - int((v - vmin) / rng * (h - 4))
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(self.surface, color, False, pts, 1)
        # Labels
        self.surface.blit(f.render(f'{label} {vmax:.4f}', True, color), (x0 + 2, y0 + 1))
        self.surface.blit(f.render(f'{vmin:.4f}', True, color), (x0 + 2, y0 + h - 10))
