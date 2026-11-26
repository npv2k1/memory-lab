export interface TopicEntity {
  id: string;
  name: string;
  type: string;
  mentions: number;
}

export interface TopicChunk {
  id: string;
  source_type: "document" | "memory";
  source_label: string;
  document_id?: string | null;
  version?: string | null;
  heading?: string;
  preview: string;
  text: string;
}

export interface TopicCluster {
  id: string;
  label: string;
  keywords: string[];
  item_count: number;
  share_pct: number;
  primary_source: string;
  sources: { label: string; count: number }[];
  top_entities: TopicEntity[];
  representative_chunks: TopicChunk[];
}

export interface TopicMapPayload {
  total_items: number;
  topic_count: number;
  topics: TopicCluster[];
}

export interface EgoNode {
  id: string;
  name: string;
  type: string;
  hop: number;
  mentions: number;
  degree: number;
  sources: string[];
  evidence_refs: string[];
  evidence_sentence: string;
}

export interface EgoEdge {
  id: string;
  source: string;
  target: string;
  predicate: string;
  predicate_label: string;
  confidence: number;
  weight: number;
  source_type: string;
  evidence_id: string;
  source_label: string;
  evidence_sentence: string;
}

export interface EgoGraphPayload {
  center: EgoNode;
  depth: number;
  nodes: EgoNode[];
  edges: EgoEdge[];
  hidden_cooccurrence_count: number;
  stats: {
    node_count: number;
    edge_count: number;
    explicit_edge_count: number;
  };
}

// Harmonious editorial source colors (muted natural inks with high-contrast text)
const SOURCE_PALETTE = [
  { bg: "#e8f3ee", border: "#b7dbc9", accent: "#146c43", ink: "#0f3d28" }, // Forest
  { bg: "#eaf1fa", border: "#bfd5f1", accent: "#1d5fa7", ink: "#13355c" }, // Slate Blue
  { bg: "#fbf0e6", border: "#ecccb0", accent: "#b45309", ink: "#5a2b07" }, // Warm Ochre
  { bg: "#f4edf7", border: "#dac4e3", accent: "#7c3f99", ink: "#3f1d50" }, // Muted Plum
  { bg: "#e8f4f5", border: "#b5dce0", accent: "#0f766e", ink: "#11423e" }, // Deep Teal
  { bg: "#f9ecec", border: "#e7c1c1", accent: "#a63232", ink: "#541818" }, // Terracotta
  { bg: "#f1f4e6", border: "#d2dcb1", accent: "#4d7c0f", ink: "#284107" }, // Olive
  { bg: "#eff1f5", border: "#cfd6e2", accent: "#475569", ink: "#1e293b" }, // Cool Slate
];

export const ENTITY_TYPE_META: Record<string, { label: string; color: string; bg: string }> = {
  person: { label: "Người", color: "#b45309", bg: "#fef3c7" },
  organization: { label: "Tổ chức", color: "#1d5fa7", bg: "#dbeafe" },
  technology: { label: "Công nghệ", color: "#146c43", bg: "#dcfce7" },
  location: { label: "Địa điểm", color: "#0f766e", bg: "#ccfbf1" },
  date: { label: "Thời gian", color: "#7c3f99", bg: "#f3e8ff" },
  concept: { label: "Khái niệm", color: "#475569", bg: "#f1f5f9" },
  other: { label: "Khác", color: "#64748b", bg: "#f1f5f9" },
};

export function entityTypeLabel(type: string): string {
  return ENTITY_TYPE_META[type]?.label || "Khái niệm";
}

function hashSource(label: string): number {
  let h = 0;
  for (let i = 0; i < label.length; i++) {
    h = (h * 31 + label.charCodeAt(i)) | 0;
  }
  return Math.abs(h) % SOURCE_PALETTE.length;
}

export function sourcePaletteFor(label: string) {
  return SOURCE_PALETTE[hashSource(label || "default")];
}

function escapeHtml(s: string): string {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

interface LayoutRect {
  x: number;
  y: number;
  w: number;
  h: number;
  topic: TopicCluster;
}

/**
 * Squarified treemap layout (Bruls, Huizing, van Wijk) in normalized [0..100] percentage coordinates.
 */
function squarifyTopics(topics: TopicCluster[], x = 0, y = 0, w = 100, h = 100): LayoutRect[] {
  const valid = topics.filter((t) => t.item_count > 0).sort((a, b) => b.item_count - a.item_count);
  if (!valid.length) return [];

  const totalValue = valid.reduce((acc, t) => acc + t.item_count, 0);
  const totalArea = w * h;
  const items = valid.map((t) => ({ topic: t, area: (t.item_count / totalValue) * totalArea }));

  const rects: LayoutRect[] = [];
  let rx = x;
  let ry = y;
  let rw = w;
  let rh = h;
  let idx = 0;

  const worstRatio = (row: { area: number }[], side: number): number => {
    if (!row.length || side <= 0) return Infinity;
    const sum = row.reduce((a, b) => a + b.area, 0);
    const maxA = Math.max(...row.map((r) => r.area));
    const minA = Math.min(...row.map((r) => r.area));
    if (minA <= 0 || sum <= 0) return Infinity;
    const s2 = side * side;
    const sum2 = sum * sum;
    return Math.max((s2 * maxA) / sum2, sum2 / (s2 * minA));
  };

  while (idx < items.length) {
    const horizontal = rw >= rh;
    const side = horizontal ? rh : rw;
    const row: { topic: TopicCluster; area: number }[] = [items[idx++]];

    while (idx < items.length) {
      const next = items[idx];
      const curWorst = worstRatio(row, side);
      const nextWorst = worstRatio([...row, next], side);
      if (nextWorst <= curWorst) {
        row.push(next);
        idx++;
      } else {
        break;
      }
    }

    const rowSum = row.reduce((a, b) => a + b.area, 0);
    if (horizontal) {
      const colW = side > 0 ? rowSum / side : rw;
      let cy = ry;
      for (const item of row) {
        const itemH = colW > 0 ? item.area / colW : rh / row.length;
        rects.push({ x: rx, y: cy, w: colW, h: itemH, topic: item.topic });
        cy += itemH;
      }
      rx += colW;
      rw = Math.max(0, rw - colW);
    } else {
      const rowH = side > 0 ? rowSum / side : rh;
      let cx = rx;
      for (const item of row) {
        const itemW = rowH > 0 ? item.area / rowH : rw / row.length;
        rects.push({ x: cx, y: ry, w: itemW, h: rowH, topic: item.topic });
        cx += itemW;
      }
      ry += rowH;
      rh = Math.max(0, rh - rowH);
    }
  }

  return rects;
}

export function renderTopicTreemap(
  container: HTMLElement,
  data: TopicMapPayload,
  selectedTopicId: string | null,
  onSelect: (topic: TopicCluster) => void
): void {
  if (!data.topics || !data.topics.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">Chưa có đủ dữ liệu để lập bản đồ chủ đề</div>
        <p class="empty-sub">Hãy tải lên ít nhất một tài liệu hoặc lưu một vài ký ức để hệ thống tự động phân nhóm chủ đề.</p>
      </div>
    `;
    return;
  }

  const totalItems = Math.max(1, data.total_items || data.topics.reduce((a, t) => a + t.item_count, 0));
  const rects = squarifyTopics(data.topics, 0, 0, 100, 100);
  const tilesHtml = rects
    .map((r) => {
      const pal = sourcePaletteFor(r.topic.primary_source);
      const isSelected = r.topic.id === selectedTopicId;
      const isCompact = r.w < 24 || r.h < 26;
      const isTiny = r.w < 16 || r.h < 18;
      const pct = r.topic.share_pct ?? Math.round((r.topic.item_count * 1000) / totalItems) / 10;
      const chips = !isCompact
        ? r.topic.top_entities
            .slice(0, 3)
            .map((e) => `<span class="tm-chip">${escapeHtml(e.name)}</span>`)
            .join("")
        : "";
      return `
        <button
          type="button"
          class="tm-tile ${isSelected ? "is-selected" : ""} ${isCompact ? "is-compact" : ""} ${isTiny ? "is-tiny" : ""}"
          data-topic-id="${escapeHtml(r.topic.id)}"
          style="
            left: calc(${r.x.toFixed(3)}% + 3px);
            top: calc(${r.y.toFixed(3)}% + 3px);
            width: calc(${r.w.toFixed(3)}% - 6px);
            height: calc(${r.h.toFixed(3)}% - 6px);
            --tm-bg: ${pal.bg};
            --tm-border: ${pal.border};
            --tm-accent: ${pal.accent};
            --tm-ink: ${pal.ink};
          "
          title="${escapeHtml(r.topic.label)} (${r.topic.item_count} đoạn · ${escapeHtml(r.topic.primary_source)})"
        >
          <div class="tm-top">
            <span class="tm-count">${r.topic.item_count} đoạn</span>
            ${!isTiny ? `<span class="tm-share">${pct}%</span>` : ""}
          </div>
          <div class="tm-label">${escapeHtml(r.topic.label)}</div>
          ${!isTiny ? `<div class="tm-source">${escapeHtml(r.topic.primary_source)}</div>` : ""}
          ${chips ? `<div class="tm-chips">${chips}</div>` : ""}
        </button>
      `;
    })
    .join("");

  container.innerHTML = `<div class="treemap-stage">${tilesHtml}</div>`;
  container.querySelectorAll<HTMLButtonElement>(".tm-tile").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tid = btn.dataset.topicId;
      const found = data.topics.find((t) => t.id === tid);
      if (found) onSelect(found);
    });
  });
}

/**
 * Interactive radial Ego Graph SVG renderer centered on 1 entity with labeled Vietnamese edges.
 */
export class EgoGraphRenderer {
  private container: HTMLElement;
  private tooltipEl: HTMLElement;
  private onFocusEntity: (entityId: string) => void;
  private onSelectNode: (node: EgoNode) => void;
  private zoom = 1;
  private panX = 0;
  private panY = 0;
  private dragging = false;
  private dragStartX = 0;
  private dragStartY = 0;
  private svgEl: SVGSVGElement | null = null;
  private viewportG: SVGGElement | null = null;

  constructor(
    container: HTMLElement,
    tooltipEl: HTMLElement,
    onFocusEntity: (entityId: string) => void,
    onSelectNode: (node: EgoNode) => void
  ) {
    this.container = container;
    this.tooltipEl = tooltipEl;
    this.onFocusEntity = onFocusEntity;
    this.onSelectNode = onSelectNode;
  }

  public resetView(): void {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
  }

  public zoomBy(factor: number): void {
    this.zoom = Math.max(0.55, Math.min(2.2, this.zoom * factor));
    this.applyTransform();
  }

  private applyTransform(): void {
    if (this.viewportG) {
      this.viewportG.setAttribute("transform", `translate(${this.panX.toFixed(1)}, ${this.panY.toFixed(1)}) scale(${this.zoom.toFixed(3)})`);
    }
  }

  public render(data: EgoGraphPayload): void {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;

    const width = Math.max(640, this.container.clientWidth || 760);
    const height = Math.max(460, this.container.clientHeight || 500);
    const cx = width / 2;
    const cy = height / 2;

    const hop1 = data.nodes.filter((n) => n.hop === 1);
    const hop2 = data.nodes.filter((n) => n.hop === 2);

    const pos = new Map<string, { x: number; y: number }>();
    pos.set(data.center.id, { x: cx, y: cy });

    const r1 = Math.min(width, height) * (hop2.length ? 0.27 : 0.34);
    const r2 = Math.min(width, height) * 0.44;

    hop1.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / Math.max(1, hop1.length) - Math.PI / 2;
      pos.set(node.id, {
        x: cx + Math.cos(angle) * r1 * 1.12,
        y: cy + Math.sin(angle) * r1 * 0.94,
      });
    });

    // Position 2-hop nodes near their connected 1-hop parent angle
    const hop1Angle = new Map<string, number>();
    hop1.forEach((node, i) => {
      hop1Angle.set(node.id, (2 * Math.PI * i) / Math.max(1, hop1.length) - Math.PI / 2);
    });
    const usedAngles: number[] = [];
    hop2.forEach((node, i) => {
      const parentEdge = data.edges.find(
        (e) =>
          (e.source === node.id && hop1Angle.has(e.target)) ||
          (e.target === node.id && hop1Angle.has(e.source))
      );
      const parentId = parentEdge
        ? parentEdge.source === node.id
          ? parentEdge.target
          : parentEdge.source
        : null;
      let angle = parentId && hop1Angle.has(parentId)
        ? hop1Angle.get(parentId)! + ((i % 3) - 1) * 0.24
        : (2 * Math.PI * i) / Math.max(1, hop2.length) - Math.PI / 3;
      while (usedAngles.some((a) => Math.abs(Math.atan2(Math.sin(a - angle), Math.cos(a - angle))) < 0.22)) {
        angle += 0.22;
      }
      usedAngles.push(angle);
      pos.set(node.id, {
        x: cx + Math.cos(angle) * r2 * 1.15,
        y: cy + Math.sin(angle) * r2 * 0.96,
      });
    });

    const svgNS = "http://www.w3.org/2000/svg";
    this.container.innerHTML = "";

    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "ego-svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");

    // Arrowhead markers
    const defs = document.createElementNS(svgNS, "defs");
    defs.innerHTML = `
      <marker id="ego-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#475569" />
      </marker>
      <marker id="ego-arrow-muted" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#94a3b8" />
      </marker>
    `;
    svg.appendChild(defs);

    const viewport = document.createElementNS(svgNS, "g");
    viewport.setAttribute("class", "ego-viewport");
    svg.appendChild(viewport);

    // Subtle orbit rings
    const ring1 = document.createElementNS(svgNS, "ellipse");
    ring1.setAttribute("cx", String(cx));
    ring1.setAttribute("cy", String(cy));
    ring1.setAttribute("rx", String(r1 * 1.12));
    ring1.setAttribute("ry", String(r1 * 0.94));
    ring1.setAttribute("class", "ego-orbit");
    viewport.appendChild(ring1);

    if (hop2.length) {
      const ring2 = document.createElementNS(svgNS, "ellipse");
      ring2.setAttribute("cx", String(cx));
      ring2.setAttribute("cy", String(cy));
      ring2.setAttribute("rx", String(r2 * 1.15));
      ring2.setAttribute("ry", String(r2 * 0.96));
      ring2.setAttribute("class", "ego-orbit ego-orbit-outer");
      viewport.appendChild(ring2);
    }

    // Track parallel edges between the same node pair to offset curvature
    const pairCount = new Map<string, number>();

    const edgesLayer = document.createElementNS(svgNS, "g");
    viewport.appendChild(edgesLayer);

    for (const edge of data.edges) {
      const p1 = pos.get(edge.source);
      const p2 = pos.get(edge.target);
      if (!p1 || !p2) continue;

      const pairKey = [edge.source, edge.target].sort().join("|");
      const seen = pairCount.get(pairKey) || 0;
      pairCount.set(pairKey, seen + 1);

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;

      const rStart = edge.source === data.center.id ? 34 : 23;
      const rEnd = edge.target === data.center.id ? 38 : 27;
      const sx = p1.x + ux * rStart;
      const sy = p1.y + uy * rStart;
      const tx = p2.x - ux * rEnd;
      const ty = p2.y - uy * rEnd;

      const curveOffset = (seen === 0 ? 14 : (seen % 2 === 1 ? -1 : 1) * (22 + seen * 14));
      const mx = (sx + tx) / 2 - uy * curveOffset;
      const my = (sy + ty) / 2 + ux * curveOffset;

      const isCooccur = edge.predicate === "co_occurs_with";
      const strokeWidth = Math.min(4.4, 1.4 + Math.log2(Math.max(1, edge.weight)) * 1.15);

      const edgeG = document.createElementNS(svgNS, "g");
      edgeG.setAttribute("class", `ego-edge-group ${isCooccur ? "is-cooccur" : ""}`);

      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", `M ${sx.toFixed(1)} ${sy.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${tx.toFixed(1)} ${ty.toFixed(1)}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", isCooccur ? "#cbd5e1" : "#64748b");
      path.setAttribute("stroke-width", strokeWidth.toFixed(2));
      if (isCooccur) {
        path.setAttribute("stroke-dasharray", "5 4");
      } else {
        path.setAttribute("marker-end", "url(#ego-arrow)");
      }
      edgeG.appendChild(path);

      // Edge label pill at quadratic midpoint t=0.5
      const lx = 0.25 * sx + 0.5 * mx + 0.25 * tx;
      const ly = 0.25 * sy + 0.5 * my + 0.25 * ty;
      const labelText = edge.weight > 1 ? `${edge.predicate_label} (${edge.weight})` : edge.predicate_label;

      const pillW = Math.max(46, labelText.length * 6.3 + 12);
      const pill = document.createElementNS(svgNS, "rect");
      pill.setAttribute("x", String(lx - pillW / 2));
      pill.setAttribute("y", String(ly - 9));
      pill.setAttribute("width", String(pillW));
      pill.setAttribute("height", "18");
      pill.setAttribute("rx", "9");
      pill.setAttribute("class", `ego-edge-pill ${isCooccur ? "is-cooccur" : ""}`);
      edgeG.appendChild(pill);

      const textEl = document.createElementNS(svgNS, "text");
      textEl.setAttribute("x", String(lx));
      textEl.setAttribute("y", String(ly + 3.5));
      textEl.setAttribute("text-anchor", "middle");
      textEl.setAttribute("class", `ego-edge-label ${isCooccur ? "is-cooccur" : ""}`);
      textEl.textContent = labelText;
      edgeG.appendChild(textEl);

      edgeG.addEventListener("mouseenter", (ev) => {
        const srcNode = data.nodes.find((n) => n.id === edge.source)?.name || edge.source;
        const tgtNode = data.nodes.find((n) => n.id === edge.target)?.name || edge.target;
        this.showTooltip(
          ev,
          `<strong>${escapeHtml(srcNode)}</strong> <span class="tt-pred">${escapeHtml(edge.predicate_label)}</span> <strong>${escapeHtml(tgtNode)}</strong>` +
            `<div class="tt-meta">Xuất hiện ${edge.weight} lần · Nguồn: ${escapeHtml(edge.source_label)}</div>` +
            (edge.evidence_sentence ? `<div class="tt-quote">“${escapeHtml(edge.evidence_sentence)}”</div>` : "")
        );
      });
      edgeG.addEventListener("mousemove", (ev) => this.moveTooltip(ev));
      edgeG.addEventListener("mouseleave", () => this.hideTooltip());

      edgesLayer.appendChild(edgeG);
    }

    // Draw nodes on top
    const nodesLayer = document.createElementNS(svgNS, "g");
    viewport.appendChild(nodesLayer);

    for (const node of data.nodes) {
      const p = pos.get(node.id);
      if (!p) continue;
      const meta = ENTITY_TYPE_META[node.type] || ENTITY_TYPE_META.concept;
      const isCenter = node.hop === 0;
      const radius = isCenter ? 30 : node.hop === 1 ? Math.min(24, 15 + Math.log2(node.mentions + 1) * 3) : 14;

      const nodeG = document.createElementNS(svgNS, "g");
      nodeG.setAttribute("class", `ego-node-group hop-${node.hop}`);
      nodeG.setAttribute("data-entity-id", node.id);
      nodeG.setAttribute("transform", `translate(${p.x.toFixed(1)}, ${p.y.toFixed(1)})`);

      if (isCenter) {
        const halo = document.createElementNS(svgNS, "circle");
        halo.setAttribute("r", String(radius + 7));
        halo.setAttribute("fill", meta.bg);
        halo.setAttribute("stroke", meta.color);
        halo.setAttribute("stroke-opacity", "0.35");
        halo.setAttribute("stroke-width", "1.5");
        nodeG.appendChild(halo);
      }

      const circle = document.createElementNS(svgNS, "circle");
      circle.setAttribute("r", String(radius));
      circle.setAttribute("fill", isCenter ? meta.color : "#ffffff");
      circle.setAttribute("stroke", meta.color);
      circle.setAttribute("stroke-width", isCenter ? "3" : "2.4");
      nodeG.appendChild(circle);

      // Initials / count inside circle
      const badge = document.createElementNS(svgNS, "text");
      badge.setAttribute("y", "4");
      badge.setAttribute("text-anchor", "middle");
      badge.setAttribute("class", isCenter ? "ego-node-badge is-center" : "ego-node-badge");
      badge.setAttribute("fill", isCenter ? "#ffffff" : meta.color);
      badge.textContent = String(node.mentions);
      nodeG.appendChild(badge);

      // Readable label below node with soft white halo
      const displayLabel = node.name.length > 26 ? node.name.slice(0, 24) + "…" : node.name;
      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("y", String(radius + 16));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", isCenter ? "ego-node-label is-center" : "ego-node-label");
      label.textContent = displayLabel;
      nodeG.appendChild(label);

      nodeG.addEventListener("mouseenter", (ev) => {
        this.showTooltip(
          ev,
          `<strong>${escapeHtml(node.name)}</strong> <span class="tt-badge" style="background:${meta.bg};color:${meta.color}">${escapeHtml(meta.label)}</span>` +
            `<div class="tt-meta">${node.mentions} lần nhắc · ${node.degree} liên kết trực tiếp</div>` +
            (node.evidence_sentence ? `<div class="tt-quote">“${escapeHtml(node.evidence_sentence)}”</div>` : "") +
            (!isCenter ? `<div class="tt-hint">Bấm để xoay đồ thị quanh thực thể này</div>` : "")
        );
      });
      nodeG.addEventListener("mousemove", (ev) => this.moveTooltip(ev));
      nodeG.addEventListener("mouseleave", () => this.hideTooltip());
      nodeG.addEventListener("click", (ev) => {
        ev.stopPropagation();
        this.hideTooltip();
        if (isCenter) {
          this.onSelectNode(node);
        } else {
          this.onFocusEntity(node.id);
        }
      });

      nodesLayer.appendChild(nodeG);
    }

    // Pan & Zoom interactions
    svg.addEventListener("mousedown", (ev) => {
      if ((ev.target as Element).closest(".ego-node-group")) return;
      this.dragging = true;
      this.dragStartX = ev.clientX - this.panX;
      this.dragStartY = ev.clientY - this.panY;
      svg.classList.add("is-dragging");
    });
    window.addEventListener("mousemove", (ev) => {
      if (!this.dragging) return;
      this.panX = ev.clientX - this.dragStartX;
      this.panY = ev.clientY - this.dragStartY;
      this.applyTransform();
    });
    window.addEventListener("mouseup", () => {
      if (!this.dragging) return;
      this.dragging = false;
      svg.classList.remove("is-dragging");
    });
    svg.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        const factor = ev.deltaY < 0 ? 1.08 : 0.92;
        this.zoomBy(factor);
      },
      { passive: false }
    );

    this.svgEl = svg;
    this.viewportG = viewport;
    this.container.appendChild(svg);
  }

  private showTooltip(ev: MouseEvent, html: string): void {
    this.tooltipEl.innerHTML = html;
    this.tooltipEl.hidden = false;
    this.moveTooltip(ev);
  }

  private moveTooltip(ev: MouseEvent): void {
    const rect = this.container.getBoundingClientRect();
    const x = Math.min(rect.width - 280, Math.max(12, ev.clientX - rect.left + 14));
    const y = Math.min(rect.height - 110, Math.max(12, ev.clientY - rect.top + 14));
    this.tooltipEl.style.left = `${x}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  private hideTooltip(): void {
    this.tooltipEl.hidden = true;
  }
}
