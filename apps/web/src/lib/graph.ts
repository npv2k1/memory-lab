export interface GraphNodeData {
  id: string;
  name: string;
  type: string;
  mentions: number;
  degree: number;
  document_count: number;
  memory_count: number;
  sources: string[];
  evidence_refs: string[];
}

export interface GraphEdgeData {
  id: string;
  source: string;
  target: string;
  predicate: string;
  confidence: number;
  source_type: string;
  evidence_id: string;
  source_label: string;
}

export interface KnowledgeGraphPayload {
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
  stats: {
    node_count: number;
    edge_count: number;
    total_nodes_available: number;
    total_edges_available: number;
    document_nodes: number;
    memory_nodes: number;
  };
}

interface SimNode extends GraphNodeData {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

const SVG_NS = "http://www.w3.org/2000/svg";

const TYPE_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  technology: { fill: "#0e7490", stroke: "#155e75", text: "#0e5a63" },
  organization: { fill: "#3b5998", stroke: "#29487a", text: "#29487a" },
  concept: { fill: "#146c43", stroke: "#0f5132", text: "#0f5132" },
  person: { fill: "#b45309", stroke: "#92400e", text: "#8a4b08" },
  date: { fill: "#a16207", stroke: "#854d0e", text: "#854d0e" },
  location: { fill: "#4f6d7a", stroke: "#36505c", text: "#36505c" },
  other: { fill: "#525866", stroke: "#374151", text: "#4b5260" },
};

function colorFor(type: string) {
  return TYPE_COLORS[type] || TYPE_COLORS.other;
}

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {}
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  return node;
}

export class InteractiveKnowledgeGraph {
  private svg: SVGSVGElement;
  private viewportG: SVGGElement;
  private edgesG: SVGGElement;
  private nodesG: SVGGElement;
  private onSelect: (node: GraphNodeData, edges: GraphEdgeData[]) => void;

  private simNodes: SimNode[] = [];
  private simEdges: GraphEdgeData[] = [];
  private nodeById = new Map<string, SimNode>();
  private selectedId: string | null = null;
  private hoveredId: string | null = null;

  private zoom = 1;
  private panX = 0;
  private panY = 0;
  private width = 820;
  private height = 520;

  private draggingNode: SimNode | null = null;
  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;

  constructor(
    svg: SVGSVGElement,
    onSelect: (node: GraphNodeData, edges: GraphEdgeData[]) => void
  ) {
    this.svg = svg;
    this.onSelect = onSelect;
    this.svg.replaceChildren();

    const defs = svgEl("defs");
    const marker = svgEl("marker", {
      id: "kb-arrow",
      viewBox: "0 -4 8 8",
      refX: 18,
      refY: 0,
      markerWidth: 6,
      markerHeight: 6,
      orient: "auto",
    });
    marker.append(svgEl("path", { d: "M0,-3.5L7,0L0,3.5", fill: "#9ca3af" }));
    defs.append(marker);
    this.svg.append(defs);

    this.viewportG = svgEl("g");
    this.edgesG = svgEl("g");
    this.nodesG = svgEl("g");
    this.viewportG.append(this.edgesG, this.nodesG);
    this.svg.append(this.viewportG);

    this.bindViewportEvents();
  }

  public zoomBy(factor: number) {
    this.zoom = Math.max(0.4, Math.min(2.6, this.zoom * factor));
    this.applyTransform();
  }

  public resetView() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
  }

  public selectNodeById(id: string) {
    const node = this.nodeById.get(id);
    if (!node) return;
    this.selectedId = id;
    const connected = this.simEdges.filter((e) => e.source === id || e.target === id);
    this.onSelect(node, connected);
    this.updateHighlights();
  }

  public render(payload: KnowledgeGraphPayload) {
    const rect = this.svg.getBoundingClientRect();
    this.width = rect.width > 200 ? rect.width : 820;
    this.height = rect.height > 200 ? rect.height : 520;
    this.svg.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);

    this.simEdges = payload.edges || [];
    const nodes = payload.nodes || [];

    this.edgesG.replaceChildren();
    this.nodesG.replaceChildren();
    this.nodeById.clear();

    if (!nodes.length) {
      const text = svgEl("text", {
        x: this.width / 2,
        y: this.height / 2,
        "text-anchor": "middle",
        fill: "#7a818f",
        "font-size": 14,
      });
      text.textContent = "Chưa có thực thể trong đồ thị. Hãy xử lý tài liệu hoặc lưu ký ức.";
      this.nodesG.append(text);
      return;
    }

    const cx = this.width / 2;
    const cy = this.height / 2;
    const radius = Math.min(this.width, this.height) * 0.34;

    this.simNodes = nodes.map((n, idx) => {
      const angle = (idx / Math.max(nodes.length, 1)) * Math.PI * 2;
      const ring = idx % 2 === 0 ? 1 : 0.68;
      const sim: SimNode = {
        ...n,
        x: cx + Math.cos(angle) * radius * ring,
        y: cy + Math.sin(angle) * radius * ring,
        vx: 0,
        vy: 0,
        r: Math.max(10, Math.min(24, 9 + Math.sqrt(n.mentions + n.degree) * 3.4)),
      };
      this.nodeById.set(n.id, sim);
      return sim;
    });

    this.runForceLayout(120);
    this.draw();
    this.applyTransform();

    if (this.selectedId && this.nodeById.has(this.selectedId)) {
      this.selectNodeById(this.selectedId);
    } else if (this.simNodes.length > 0) {
      this.selectNodeById(this.simNodes[0].id);
    }
  }

  private runForceLayout(iterations: number) {
    const cx = this.width / 2;
    const cy = this.height / 2;
    const repulsion = 6200;
    const idealLen = 152;

    for (let step = 0; step < iterations; step++) {
      // Repulsion between all node pairs
      for (let i = 0; i < this.simNodes.length; i++) {
        const a = this.simNodes[i];
        for (let j = i + 1; j < this.simNodes.length; j++) {
          const b = this.simNodes[j];
          const dx = b.x - a.x || 0.01;
          const dy = b.y - a.y || 0.01;
          const distSq = dx * dx + dy * dy + 36;
          const dist = Math.sqrt(distSq);
          const force = repulsion / distSq;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }

      // Spring attraction along edges
      for (const edge of this.simEdges) {
        const a = this.nodeById.get(edge.source);
        const b = this.nodeById.get(edge.target);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const spring = (dist - idealLen) * 0.04;
        const fx = (dx / dist) * spring;
        const fy = (dy / dist) * spring;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }

      // Gentle gravity toward center + damping + bounds
      for (const node of this.simNodes) {
        node.vx += (cx - node.x) * 0.011;
        node.vy += (cy - node.y) * 0.011;
        node.vx *= 0.8;
        node.vy *= 0.8;
        node.x = Math.max(54, Math.min(this.width - 54, node.x + node.vx));
        node.y = Math.max(46, Math.min(this.height - 46, node.y + node.vy));
      }
    }
  }

  private draw() {
    this.edgesG.replaceChildren();
    this.nodesG.replaceChildren();

    for (const edge of this.simEdges) {
      const a = this.nodeById.get(edge.source);
      const b = this.nodeById.get(edge.target);
      if (!a || !b) continue;

      const g = svgEl("g", {
        "data-edge-source": edge.source,
        "data-edge-target": edge.target,
      });
      const isCo = edge.predicate === "co_occurs_with";
      const line = svgEl("line", {
        x1: a.x,
        y1: a.y,
        x2: b.x,
        y2: b.y,
        stroke: isCo ? "#cbd2d9" : "#7c9487",
        "stroke-width": isCo ? 1.2 : 1.9,
        "stroke-dasharray": isCo ? "4 3" : "",
        "marker-end": isCo ? "" : "url(#kb-arrow)",
      });

      const label = svgEl("text", {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2 - 4,
        "text-anchor": "middle",
        fill: "#525866",
        "font-size": 10,
        "font-family": "var(--font-mono)",
        opacity: isCo ? 0 : 0.85,
      });
      label.textContent = edge.predicate;

      g.append(line, label);
      this.edgesG.append(g);
    }

    for (const node of this.simNodes) {
      const palette = colorFor(node.type);
      const g = svgEl("g", {
        "data-node-id": node.id,
        transform: `translate(${node.x}, ${node.y})`,
        style: "cursor:pointer",
      });

      const halo = svgEl("circle", {
        r: node.r + 4,
        fill: palette.fill,
        opacity: 0.14,
        class: "node-halo",
      });

      const circle = svgEl("circle", {
        r: node.r,
        fill: palette.fill,
        stroke: "#ffffff",
        "stroke-width": 2.2,
      });

      const label = svgEl("text", {
        y: node.r + 14,
        "text-anchor": "middle",
        fill: "#14171a",
        "font-size": 11.5,
        "font-weight": 600,
      });
      label.textContent = node.name.length > 22 ? node.name.slice(0, 20) + "…" : node.name;

      g.append(halo, circle, label);

      g.addEventListener("mouseenter", () => {
        this.hoveredId = node.id;
        this.updateHighlights();
      });
      g.addEventListener("mouseleave", () => {
        this.hoveredId = null;
        this.updateHighlights();
      });
      g.addEventListener("mousedown", (ev) => {
        ev.stopPropagation();
        this.draggingNode = node;
        this.selectNodeById(node.id);
      });

      this.nodesG.append(g);
    }
    this.updateHighlights();
  }

  private updatePositions() {
    for (const g of Array.from(this.nodesG.children)) {
      const id = g.getAttribute("data-node-id");
      if (!id) continue;
      const n = this.nodeById.get(id);
      if (n) g.setAttribute("transform", `translate(${n.x}, ${n.y})`);
    }
    for (const g of Array.from(this.edgesG.children)) {
      const s = g.getAttribute("data-edge-source");
      const t = g.getAttribute("data-edge-target");
      if (!s || !t) continue;
      const a = this.nodeById.get(s);
      const b = this.nodeById.get(t);
      if (!a || !b) continue;
      const line = g.querySelector("line");
      const text = g.querySelector("text");
      if (line) {
        line.setAttribute("x1", String(a.x));
        line.setAttribute("y1", String(a.y));
        line.setAttribute("x2", String(b.x));
        line.setAttribute("y2", String(b.y));
      }
      if (text) {
        text.setAttribute("x", String((a.x + b.x) / 2));
        text.setAttribute("y", String((a.y + b.y) / 2 - 4));
      }
    }
  }

  private updateHighlights() {
    const focus = this.hoveredId || this.selectedId;
    const neighbors = new Set<string>();
    if (focus) {
      neighbors.add(focus);
      for (const e of this.simEdges) {
        if (e.source === focus) neighbors.add(e.target);
        if (e.target === focus) neighbors.add(e.source);
      }
    }

    for (const g of Array.from(this.nodesG.children)) {
      const id = g.getAttribute("data-node-id");
      if (!id) continue;
      const isSelected = id === this.selectedId;
      const isNeighbor = !focus || neighbors.has(id);
      g.setAttribute("opacity", isNeighbor ? "1" : "0.24");
      const halo = g.querySelector(".node-halo");
      if (halo) halo.setAttribute("opacity", isSelected ? "0.35" : "0.12");
    }

    for (const g of Array.from(this.edgesG.children)) {
      const s = g.getAttribute("data-edge-source");
      const t = g.getAttribute("data-edge-target");
      const active = !focus || s === focus || t === focus;
      g.setAttribute("opacity", active ? "1" : "0.14");
      const text = g.querySelector("text");
      if (text) {
        const isCo = text.textContent === "co_occurs_with";
        text.setAttribute("opacity", !isCo && active ? "0.95" : "0");
      }
    }
  }

  private applyTransform() {
    const cx = this.width / 2;
    const cy = this.height / 2;
    this.viewportG.setAttribute(
      "transform",
      `translate(${this.panX + cx * (1 - this.zoom)}, ${this.panY + cy * (1 - this.zoom)}) scale(${this.zoom})`
    );
  }

  private screenToGraph(clientX: number, clientY: number) {
    const rect = this.svg.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * this.width;
    const sy = ((clientY - rect.top) / rect.height) * this.height;
    const cx = this.width / 2;
    const cy = this.height / 2;
    return {
      x: (sx - this.panX - cx * (1 - this.zoom)) / this.zoom,
      y: (sy - this.panY - cy * (1 - this.zoom)) / this.zoom,
    };
  }

  private bindViewportEvents() {
    this.svg.addEventListener("mousedown", (ev) => {
      if (this.draggingNode) return;
      this.isPanning = true;
      this.panStartX = ev.clientX - this.panX;
      this.panStartY = ev.clientY - this.panY;
    });

    window.addEventListener("mousemove", (ev) => {
      if (this.draggingNode) {
        const pt = this.screenToGraph(ev.clientX, ev.clientY);
        this.draggingNode.x = pt.x;
        this.draggingNode.y = pt.y;
        this.updatePositions();
      } else if (this.isPanning) {
        this.panX = ev.clientX - this.panStartX;
        this.panY = ev.clientY - this.panStartY;
        this.applyTransform();
      }
    });

    window.addEventListener("mouseup", () => {
      this.draggingNode = null;
      this.isPanning = false;
    });

    this.svg.addEventListener(
      "wheel",
      (ev) => {
        ev.preventDefault();
        this.zoomBy(ev.deltaY < 0 ? 1.1 : 0.9);
      },
      { passive: false }
    );
  }
}
