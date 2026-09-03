/**
 * Before/after comparison.
 *
 * Cesium can only hold one terrain provider at a time, so a true terrain swipe is not
 * available. The architecture note's solution is used instead: compare *imagery*
 * hillshades with `ImageryLayer.splitDirection` and `scene.splitPosition`, which gives
 * a real side-by-side of the bicubic-GTDR relief against the learned relief, and swap
 * the terrain provider with a separate toggle for the 3-D read.
 */
import { SplitDirection, type ImageryLayer, type Viewer } from "cesium";

export interface SwipePair {
  left: ImageryLayer;
  right: ImageryLayer;
}

export function enableSwipe(viewer: Viewer, pair: SwipePair, position = 0.5): void {
  pair.left.splitDirection = SplitDirection.LEFT;
  pair.right.splitDirection = SplitDirection.RIGHT;
  pair.left.show = true;
  pair.right.show = true;
  viewer.scene.splitPosition = position;
}

export function disableSwipe(pair: SwipePair): void {
  pair.left.splitDirection = SplitDirection.NONE;
  pair.right.splitDirection = SplitDirection.NONE;
}

export function setSplitPosition(viewer: Viewer, fraction: number): void {
  viewer.scene.splitPosition = Math.min(0.99, Math.max(0.01, fraction));
}

/** Drag handler for a vertical splitter element overlaying the canvas. */
export function attachSplitterDrag(viewer: Viewer, handle: HTMLElement): () => void {
  let dragging = false;
  const onDown = () => (dragging = true);
  const onUp = () => (dragging = false);
  const onMove = (e: PointerEvent) => {
    if (!dragging) return;
    const rect = viewer.canvas.getBoundingClientRect();
    setSplitPosition(viewer, (e.clientX - rect.left) / rect.width);
    handle.style.left = `${viewer.scene.splitPosition * 100}%`;
  };
  handle.addEventListener("pointerdown", onDown);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointermove", onMove);
  return () => {
    handle.removeEventListener("pointerdown", onDown);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointermove", onMove);
  };
}
