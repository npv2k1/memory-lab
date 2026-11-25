export interface AtlasNeighbor {
  id: string;
  title: string;
  score: number;
}

export interface AtlasPoint {
  id: string;
  ref_id: string;
  source_type: "document" | "memory";
  title: string;
  text: string;
  preview: string;
  strategy: string;
  session_id: string;
  date: string;
  entities: string[];
  keywords: string[];
  tags: string[];
  block_ids: string[];
  source_url: string | null;
  x: number;
  y: number;
  cluster_id: number;
  cluster_label: string;
  has_dense_vector: boolean;
  neighbors: AtlasNeighbor[];
}

export interface AtlasCluster {
  id: number;
  label: string;
  keywords: string[];
  count: number;
  x: number;
  y: number;
}

export interface AtlasPayload {
  source: string;
  projection: string;
  used_dense_embeddings: boolean;
  total_points: number;
  points: AtlasPoint[];
  clusters: AtlasCluster[];
  density: {
    grid_size: number;
    bound: number;
    values: number[];
  };
  facets: {
    sources: { label: string; count: number }[];
    clusters: { id: number; label: string; count: number }[];
    strategies: { label: string; count: number }[];
    dates: { label: string; count: number }[];
    entities: { label: string; count: number }[];
  };
}

export type AtlasColorBy = "cluster" | "source_type" | "strategy" | "session_id";
export type AtlasInteractionMode = "pan" | "brush";

export interface AtlasFilterState {
  clusterId: number | null;
  sourceType: string | null;
  strategy: string | null;
  date: string | null;
  entity: string | null;
  query: string;
  brushIds: Set<string> | null;
}

const PALETTE = [
  "#0f766e", // Botanical Teal
  "#2563eb", // Ocean Blue
  "#d97706", // Warm Amber
  "#be123c", // Crimson Rose
  "#4f46e5", // Indigo
  "#4d7c0f", // Olive Leaf
  "#0891b2", // Cyan
  "#c2410c", // Terracotta
  "#475569", // Slate
];

function hashIndex(text: string): number {
  let h = 0;
  for (let i = 0; i < text.length; i++) {
    h = (h * 31 + text.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % PALETTE.length;
}

export function atlasColorForPoint(point: AtlasPoint, colorBy: AtlasColorBy): string {
  if (colorBy === "cluster") return PALETTE[point.cluster_id % PALETTE.length];
  if (colorBy === "source_type") return point.source_type === "document" ? "#0f766e" : "#2563eb";
  if (colorBy === "strategy") {
    const map: Record<string, string> = {
      document_chunk: "#0f766e",
      hybrid: "#2563eb",
      zero_token: "#d97706",
      graph: "#4d7c0f",
      vector: "#0891b2",
    };
    return map[point.strategy] || PALETTE[hashIndex(point.strategy)];
  }
  return PALETTE[hashIndex(point.session_id)];
}

export function clusterColor(clusterId: number): string {
  return PALETTE[clusterId % PALETTE.length];
}

export class EmbeddingAtlasCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private tooltipEl: HTMLElement;
  private onSelectPoint: (point: AtlasPoint | null) => void;
  private onBrushChange: (ids: Set<string> | null) => void;
  private onClusterClick: (clusterId: number) => void;

  private payload: AtlasPayload | null = null;
  private activeIds = new Set<string>();
  private searchMatchedIds = new Set<string>();
  private selectedId: string | null = null;
  private hoveredId: string | null = null;

  private colorBy: AtlasColorBy = "cluster";
  private showContours = true;
  private showLabels = true;
  private mode: AtlasInteractionMode = "pan";

  private width = 760;
  private height = 540;
  private zoom = 1;
  private panX = 0;
  private panY = 0;

  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;

  private isBrushing = false;
  private brushStart: { x: number; y: number } | null = null;
  private brushCurr: { x: number; y: number } | null = null;

  private labelBoxes: { clusterId: number; x: number; y: number; w: number; h: number }[] = [];

  constructor(
    canvas: HTMLCanvasElement,
    tooltipEl: HTMLElement,
    callbacks: {
      onSelectPoint: (point: AtlasPoint | null) => void;
      onBrushChange: (ids: Set<string> | null) => void;
      onClusterClick: (clusterId: number) => void;
    }
  ) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.tooltipEl = tooltipEl;
    this.onSelectPoint = callbacks.onSelectPoint;
    this.onBrushChange = callbacks.onBrushChange;
    this.onClusterClick = callbacks.onClusterClick;
    this.bindEvents();
  }

  public setData(payload: AtlasPayload) {
    this.payload = payload;
    this.activeIds = new Set(payload.points.map((p) => p.id));
    this.searchMatchedIds.clear();
    if (this.selectedId && !payload.points.some((p) => p.id === this.selectedId)) {
      this.selectedId = null;
    }
    if (!this.selectedId && payload.points.length > 0) {
      this.selectedId = payload.points[0].id;
      this.onSelectPoint(payload.points[0]);
    }
    this.resize();
    this.draw();
  }

  public setFilterMask(activeIds: Set<string>, searchMatchedIds?: Set<string>) {
    this.activeIds = activeIds;
    if (searchMatchedIds) this.searchMatchedIds = searchMatchedIds;
    this.draw();
  }

  public setOptions(opts: {
    colorBy?: AtlasColorBy;
    showContours?: boolean;
    showLabels?: boolean;
    mode?: AtlasInteractionMode;
  }) {
    if (opts.colorBy !== undefined) this.colorBy = opts.colorBy;
    if (opts.showContours !== undefined) this.showContours = opts.showContours;
    if (opts.showLabels !== undefined) this.showLabels = opts.showLabels;
    if (opts.mode !== undefined) {
      this.mode = opts.mode;
      this.canvas.style.cursor = this.mode === "brush" ? "crosshair" : "grab";
    }
    this.draw();
  }

  public selectPointById(id: string) {
    if (!this.payload) return;
    const pt = this.payload.points.find((p) => p.id === id) || null;
    this.selectedId = pt ? pt.id : null;
    this.onSelectPoint(pt);
    this.draw();
  }

  public zoomBy(factor: number) {
    this.zoom = Math.max(0.5, Math.min(6.0, this.zoom * factor));
    this.draw();
  }

  public resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.brushStart = null;
    this.brushCurr = null;
    this.draw();
  }

  public resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.width = rect.width > 120 ? rect.width : 760;
    this.height = rect.height > 120 ? rect.height : 540;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private worldToScreen(wx: number, wy: number): { x: number; y: number } {
    const scale = Math.min(this.width, this.height) * 0.4 * this.zoom;
    return {
      x: this.width / 2 + this.panX + wx * scale,
      y: this.height / 2 + this.panY - wy * scale,
    };
  }

  private draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Subtle scientific paper backdrop
    ctx.fillStyle = "#fbfbf9";
    ctx.fillRect(0, 0, this.width, this.height);

    // Subtle axis crosshairs at origin
    const origin = this.worldToScreen(0, 0);
    ctx.save();
    ctx.strokeStyle = "rgba(15, 23, 42, 0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, origin.y);
    ctx.lineTo(this.width, origin.y);
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, this.height);
    ctx.stroke();
    ctx.restore();

    if (!this.payload || !this.payload.points.length) {
      ctx.fillStyle = "#7a818f";
      ctx.font = "500 14px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "Chưa có điểm embedding nào. Hãy xử lý tài liệu hoặc lưu ký ức để dựng Embedding Atlas.",
        this.width / 2,
        this.height / 2
      );
      return;
    }

    if (this.showContours) {
      this.drawDensityContours();
    }

    this.drawKnnRays();
    this.drawPoints();

    if (this.showLabels) {
      this.drawClusterLabels();
    }

    if (this.isBrushing && this.brushStart && this.brushCurr) {
      const bx = Math.min(this.brushStart.x, this.brushCurr.x);
      const by = Math.min(this.brushStart.y, this.brushCurr.y);
      const bw = Math.abs(this.brushCurr.x - this.brushStart.x);
      const bh = Math.abs(this.brushCurr.y - this.brushStart.y);
      ctx.save();
      ctx.fillStyle = "rgba(20, 108, 67, 0.10)";
      ctx.strokeStyle = "#146c43";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.restore();
    }
  }

  private drawDensityContours() {
    if (!this.payload) return;
    const { grid_size: g, bound, values } = this.payload.density;
    if (!values || values.length !== g * g) return;

    const ctx = this.ctx;
    const step = (2 * bound) / (g - 1);
    const levels = [
      { threshold: 0.16, fill: "rgba(15, 118, 110, 0.045)", stroke: "rgba(15, 118, 110, 0.14)" },
      { threshold: 0.34, fill: "rgba(15, 118, 110, 0.065)", stroke: "rgba(15, 118, 110, 0.20)" },
      { threshold: 0.56, fill: "rgba(15, 118, 110, 0.085)", stroke: "rgba(15, 118, 110, 0.27)" },
      { threshold: 0.78, fill: "rgba(15, 118, 110, 0.105)", stroke: "rgba(15, 118, 110, 0.34)" },
    ];

    const topLeft = this.worldToScreen(-bound, bound);
    const bottomRight = this.worldToScreen(bound, -bound);
    const cellW = (bottomRight.x - topLeft.x) / (g - 1);
    const cellH = (bottomRight.y - topLeft.y) / (g - 1);

    ctx.save();
    // 1. Soft topographic density band fills
    for (let row = 0; row < g - 1; row++) {
      for (let col = 0; col < g - 1; col++) {
        const v00 = values[row * g + col];
        const v10 = values[row * g + col + 1];
        const v01 = values[(row + 1) * g + col];
        const v11 = values[(row + 1) * g + col + 1];
        const avg = (v00 + v10 + v01 + v11) * 0.25;
        if (avg < levels[0].threshold) continue;
        let fill = levels[0].fill;
        for (const lvl of levels) {
          if (avg >= lvl.threshold) fill = lvl.fill;
        }
        const wx = -bound + col * step;
        const wy = -bound + row * step;
        const p = this.worldToScreen(wx, wy);
        ctx.fillStyle = fill;
        ctx.fillRect(p.x, p.y - cellH, cellW + 0.8, cellH + 0.8);
      }
    }

    // 2. Crisp marching-squares isoline contour strokes
    const interp = (a: number, b: number, t: number) => (Math.abs(b - a) < 1e-6 ? 0.5 : (t - a) / (b - a));
    for (const lvl of levels) {
      ctx.strokeStyle = lvl.stroke;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let row = 0; row < g - 1; row++) {
        for (let col = 0; col < g - 1; col++) {
          const v00 = values[row * g + col];
          const v10 = values[row * g + col + 1];
          const v01 = values[(row + 1) * g + col];
          const v11 = values[(row + 1) * g + col + 1];
          const t = lvl.threshold;
          const pts: { x: number; y: number }[] = [];
          const wx0 = -bound + col * step;
          const wy0 = -bound + row * step;
          if ((v00 >= t) !== (v10 >= t)) pts.push(this.worldToScreen(wx0 + interp(v00, v10, t) * step, wy0));
          if ((v10 >= t) !== (v11 >= t)) pts.push(this.worldToScreen(wx0 + step, wy0 + interp(v10, v11, t) * step));
          if ((v01 >= t) !== (v11 >= t)) pts.push(this.worldToScreen(wx0 + interp(v01, v11, t) * step, wy0 + step));
          if ((v00 >= t) !== (v01 >= t)) pts.push(this.worldToScreen(wx0, wy0 + interp(v00, v01, t) * step));
          if (pts.length >= 2) {
            ctx.moveTo(pts[0].x, pts[0].y);
            ctx.lineTo(pts[1].x, pts[1].y);
          }
        }
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawKnnRays() {
    if (!this.payload) return;
    const focusId = this.hoveredId || this.selectedId;
    if (!focusId) return;
    const focus = this.payload.points.find((p) => p.id === focusId);
    if (!focus || !focus.neighbors.length) return;

    const byId = new Map(this.payload.points.map((p) => [p.id, p]));
    const origin = this.worldToScreen(focus.x, focus.y);
    const ctx = this.ctx;
    ctx.save();
    for (const nbr of focus.neighbors.slice(0, 5)) {
      const target = byId.get(nbr.id);
      if (!target) continue;
      const dest = this.worldToScreen(target.x, target.y);
      ctx.strokeStyle = `rgba(20, 108, 67, ${0.22 + nbr.score * 0.55})`;
      ctx.lineWidth = 1.2 + nbr.score * 1.3;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(dest.x, dest.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawPoints() {
    if (!this.payload) return;
    const ctx = this.ctx;
    const points = this.payload.points;

    // Draw inactive points first, active points on top
    const ordered = [
      ...points.filter((p) => !this.activeIds.has(p.id)),
      ...points.filter((p) => this.activeIds.has(p.id)),
    ];

    for (const pt of ordered) {
      const pos = this.worldToScreen(pt.x, pt.y);
      const isActive = this.activeIds.has(pt.id);
      const isSelected = pt.id === this.selectedId;
      const isHovered = pt.id === this.hoveredId;
      const isSearchHit = this.searchMatchedIds.has(pt.id);
      const color = atlasColorForPoint(pt, this.colorBy);
      const radius = isSelected ? 9 : isHovered ? 8.5 : isSearchHit ? 8 : 6.2;

      ctx.save();
      if (isSearchHit || isSelected) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 5.5, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? "rgba(20, 108, 67, 0.22)" : "rgba(217, 119, 6, 0.24)";
        ctx.fill();
      }

      ctx.globalAlpha = isActive ? 0.92 : 0.16;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.lineWidth = isSelected ? 2.4 : 1.4;
      ctx.strokeStyle = isSelected ? "#14171a" : "#ffffff";
      ctx.stroke();
      ctx.restore();
    }
  }

  private drawClusterLabels() {
    if (!this.payload) return;
    const ctx = this.ctx;
    this.labelBoxes = [];

    ctx.save();
    ctx.font = "600 11.5px 'Plus Jakarta Sans', Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const cluster of this.payload.clusters) {
      const pos = this.worldToScreen(cluster.x, cluster.y);
      const text = `${cluster.label} (${cluster.count})`;
      const metrics = ctx.measureText(text);
      const padX = 9;
      const w = metrics.width + padX * 2;
      const h = 22;
      let bx = pos.x - w / 2;
      let by = pos.y - h / 2 - 18;

      // Simple vertical nudge to prevent overlapping cluster pills
      for (const prev of this.labelBoxes) {
        if (bx < prev.x + prev.w + 4 && bx + w + 4 > prev.x && by < prev.y + prev.h + 4 && by + h + 4 > prev.y) {
          by = prev.y + prev.h + 6;
        }
      }

      bx = Math.max(8, Math.min(this.width - w - 8, bx));
      by = Math.max(8, Math.min(this.height - h - 8, by));
      this.labelBoxes.push({ clusterId: cluster.id, x: bx, y: by, w, h });

      ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
      ctx.strokeStyle = clusterColor(cluster.id);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.roundRect(bx, by, w, h, 999);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#14171a";
      ctx.fillText(text, bx + w / 2, by + h / 2 + 0.5);
    }
    ctx.restore();
  }

  private hitTestPoint(sx: number, sy: number): AtlasPoint | null {
    if (!this.payload) return null;
    let best: AtlasPoint | null = null;
    let bestDist = 14;
    for (const pt of this.payload.points) {
      const pos = this.worldToScreen(pt.x, pt.y);
      const d = Math.hypot(pos.x - sx, pos.y - sy);
      if (d < bestDist) {
        bestDist = d;
        best = pt;
      }
    }
    return best;
  }

  private hitTestLabel(sx: number, sy: number): number | null {
    for (const box of this.labelBoxes) {
      if (sx >= box.x && sx <= box.x + box.w && sy >= box.y && sy <= box.y + box.h) {
        return box.clusterId;
      }
    }
    return null;
  }

  private updateTooltip(pt: AtlasPoint | null, sx: number, sy: number) {
    if (!pt) {
      this.tooltipEl.style.display = "none";
      return;
    }
    this.tooltipEl.style.display = "block";
    this.tooltipEl.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = pt.title;
    const sub = document.createElement("div");
    sub.className = "atlas-tt-meta";
    sub.textContent = `${pt.source_type} · ${pt.cluster_label} · ${pt.date}`;
    const body = document.createElement("p");
    body.textContent = pt.preview;
    this.tooltipEl.append(title, sub, body);

    const left = Math.min(this.width - 280, Math.max(10, sx + 14));
    const top = Math.min(this.height - 120, Math.max(10, sy + 14));
    this.tooltipEl.style.left = `${left}px`;
    this.tooltipEl.style.top = `${top}px`;
  }

  private bindEvents() {
    this.canvas.addEventListener("mousedown", (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;

      if (this.mode === "brush") {
        this.isBrushing = true;
        this.brushStart = { x: sx, y: sy };
        this.brushCurr = { x: sx, y: sy };
        this.draw();
        return;
      }

      const hitPt = this.hitTestPoint(sx, sy);
      if (hitPt) {
        this.selectedId = hitPt.id;
        this.onSelectPoint(hitPt);
        this.draw();
        return;
      }

      const hitCluster = this.hitTestLabel(sx, sy);
      if (hitCluster !== null) {
        this.onClusterClick(hitCluster);
        return;
      }

      this.isPanning = true;
      this.panStartX = ev.clientX - this.panX;
      this.panStartY = ev.clientY - this.panY;
    });

    this.canvas.addEventListener("mousemove", (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;

      if (this.isBrushing && this.brushStart) {
        this.brushCurr = { x: sx, y: sy };
        this.draw();
        return;
      }

      if (this.isPanning) {
        this.panX = ev.clientX - this.panStartX;
        this.panY = ev.clientY - this.panStartY;
        this.draw();
        return;
      }

      const hit = this.hitTestPoint(sx, sy);
      const nextId = hit ? hit.id : null;
      if (nextId !== this.hoveredId) {
        this.hoveredId = nextId;
        this.draw();
      }
      this.updateTooltip(hit, sx, sy);
    });

    window.addEventListener("mouseup", () => {
      if (this.isBrushing && this.brushStart && this.brushCurr && this.payload) {
        const minX = Math.min(this.brushStart.x, this.brushCurr.x);
        const maxX = Math.max(this.brushStart.x, this.brushCurr.x);
        const minY = Math.min(this.brushStart.y, this.brushCurr.y);
        const maxY = Math.max(this.brushStart.y, this.brushCurr.y);
        if (maxX - minX > 8 && maxY - minY > 8) {
          const picked = new Set<string>();
          for (const pt of this.payload.points) {
            const pos = this.worldToScreen(pt.x, pt.y);
            if (pos.x >= minX && pos.x <= maxX && pos.y >= minY && pos.y <= maxY) {
              picked.add(pt.id);
            }
          }
          this.onBrushChange(picked.size > 0 ? picked : null);
        } else {
          this.onBrushChange(null);
        }
        this.isBrushing = false;
        this.brushStart = null;
        this.brushCurr = null;
        this.draw();
      }
      this.isPanning = false;
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.hoveredId = null;
      this.updateTooltip(null, 0, 0);
      this.draw();
    });

    this.canvas.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        this.zoomBy(ev.deltaY < 0 ? 1.12 : 0.89);
      },
      { passive: false }
    );
  }
}
