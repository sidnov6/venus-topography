"""Quantized-mesh terrain tiles encoded on the Venus sphere. This is Route A.

Cesium Terrain Builder and its forks hard-code WGS84 when they compute a tile's bounding
sphere and horizon occlusion point. Those fields are culling metadata: with wrong values
the heights still decode correctly and the mesh is geometrically right, but tiles vanish
at the limb or refuse to refine, and the failure reads as a renderer bug rather than a
data bug. On a sphere the two quantities are a handful of lines, so encoding them
ourselves is cheaper than debugging a tiler that disagrees with the planet.

Format (per the Cesium quantized-mesh-1.0 spec):

    header      : centerX/Y/Z (f64), minHeight/maxHeight (f32),
                  boundingSphere centerX/Y/Z + radius (f64),
                  horizonOcclusionPoint X/Y/Z (f64)
    vertexData  : vertexCount (u32), then u, v, height as zig-zag delta-encoded u16
    indexData   : triangleCount (u32), then indices, high-water-mark encoded
    edgeIndices : west, south, east, north vertex lists (u32 count + indices)

`u`, `v` and `height` are 16-bit, so height quantisation is
`(maxHeight - minHeight) / 32767`. For a Venus tile spanning 11 km of relief that is
0.34 m, which is far below the model's own uncertainty — but for a tile that is nearly
flat it is far finer, which is the point of per-tile height bounds.

Only numpy is required.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VENUS_RADIUS_M = 6_051_800.0

QUANTIZED_MAX = 32767


# --------------------------------------------------------------------------------------
# Geometry on a sphere
# --------------------------------------------------------------------------------------
def geodetic_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, height_m: np.ndarray,
                     radius_m: float = VENUS_RADIUS_M) -> np.ndarray:
    """Planetocentric lon/lat/height to body-fixed Cartesian, shape `(..., 3)`.

    On a sphere this is the textbook conversion; there is no prime-vertical radius to
    compute, which is precisely why a WGS84 tiler's version is wrong here rather than
    merely imprecise.
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    r = radius_m + np.asarray(height_m, dtype=np.float64)
    cl = np.cos(lat)
    return np.stack([r * cl * np.cos(lon), r * cl * np.sin(lon), r * np.sin(lat)], axis=-1)


def _ritter_sphere(p: np.ndarray) -> tuple[np.ndarray, float]:
    x = p[np.argmin(p[:, 0])]
    y = p[np.argmax(np.sum((p - x) ** 2, axis=1))]
    z = p[np.argmax(np.sum((p - y) ** 2, axis=1))]
    centre = (y + z) / 2.0
    radius = float(np.linalg.norm(z - y) / 2.0)
    for _ in range(3):
        d = np.linalg.norm(p - centre, axis=1)
        far = int(np.argmax(d))
        if d[far] <= radius:
            break
        new_radius = (radius + float(d[far])) / 2.0
        centre = centre + (new_radius - radius) * (p[far] - centre) / float(d[far])
        radius = new_radius
    return centre, radius


def bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """A sphere containing every vertex, as tight as is cheap.

    Cesium uses this for culling and for screen-space-error refinement. It only has to
    *contain* the tile — a loose sphere costs some over-refinement, a tight-but-wrong one
    drops geometry — but a tile at a low zoom level is a large spherical cap, and Ritter's
    algorithm alone over-estimates such caps by ~40%. Taking the better of Ritter and a
    centroid sphere recovers most of that for two extra passes over the vertices.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    candidates = [_ritter_sphere(p)]
    centroid = p.mean(axis=0)
    candidates.append((centroid, float(np.max(np.linalg.norm(p - centroid, axis=1)))))
    # A spherical cap's minimal sphere is centred on the chord axis; the midpoint of the
    # cap's extreme radial extent is a good third guess.
    axis = centroid / np.linalg.norm(centroid)
    proj = p @ axis
    mid = axis * float((proj.min() + proj.max()) / 2.0)
    candidates.append((mid, float(np.max(np.linalg.norm(p - mid, axis=1)))))

    centre, radius = min(candidates, key=lambda cr: cr[1])
    radius = max(radius, float(np.max(np.linalg.norm(p - centre, axis=1))))
    return centre, radius * (1.0 + 1e-9)


def horizon_occlusion_point(points: np.ndarray, centre_direction: np.ndarray,
                            radius_m: float = VENUS_RADIUS_M) -> np.ndarray:
    """The occlusion point in the ellipsoid-scaled space Cesium uses.

    Cesium culls a tile when this point is below the horizon. The computation scales
    every position by the inverse ellipsoid radii; on a sphere all three are equal, so
    the scaling is a single division by the radius — and using WGS84's three different
    radii here is exactly the error that makes tiles disappear at the limb.

    Follows the reference construction: for each scaled vertex, the magnitude along the
    scaled tile-centre direction at which that vertex would just be occluded; the answer
    is the maximum over vertices.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3) / radius_m
    d = np.asarray(centre_direction, dtype=np.float64) / radius_m
    d = d / np.linalg.norm(d)

    dot = p @ d
    mag_sq = np.sum(p * p, axis=1)
    mag = np.sqrt(mag_sq)

    # Reject vertices inside the unit sphere: they can never occlude anything.
    ok = mag_sq > 1.0
    if not np.any(ok):
        return d  # degenerate tile at the surface; the direction itself is safe

    dot, mag_sq, mag = dot[ok], mag_sq[ok], mag[ok]
    cos_alpha = dot / mag
    sin_alpha = np.sqrt(np.maximum(mag_sq - 1.0, 0.0)) / mag
    cos_beta = 1.0 / mag
    sin_beta = np.sqrt(np.maximum(mag_sq - 1.0, 0.0)) / mag
    denom = cos_alpha * cos_beta - sin_alpha * sin_beta
    # denom <= 0 means the vertex is on the far side and imposes no constraint.
    valid = denom > 1e-12
    if not np.any(valid):
        return d
    return d * float(np.max(1.0 / denom[valid]))


# --------------------------------------------------------------------------------------
# Encoding primitives
# --------------------------------------------------------------------------------------
def zigzag_encode(values: np.ndarray) -> np.ndarray:
    """Map signed deltas to unsigned so small negatives stay small."""
    v = np.asarray(values, dtype=np.int32)
    return ((v << 1) ^ (v >> 31)).astype(np.uint16)


def zigzag_decode(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.uint32)
    return ((v >> 1).astype(np.int32) ^ -(v & 1).astype(np.int32)).astype(np.int32)


def delta_zigzag(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.int32)
    return zigzag_encode(np.diff(v, prepend=np.int32(0)))


def undelta_zigzag(encoded: np.ndarray) -> np.ndarray:
    return np.cumsum(zigzag_decode(encoded), dtype=np.int32)


def encode_indices(indices: np.ndarray) -> np.ndarray:
    """High-water-mark encoding: each index is stored as the gap below the highest index
    seen so far, which keeps the codes tiny and 16-bit-friendly.

    The encoding is only valid when vertices are first referenced in increasing order —
    vertex `k` must make its first appearance before vertex `k + 1`. Otherwise the code
    goes negative, wraps in the unsigned buffer, and the mesh decodes to garbage without
    any error. `reorder_for_encoding` establishes that ordering; the assertion here is
    what turns a silent corruption into a failure.
    """
    idx = np.asarray(indices, dtype=np.int64)
    out = np.empty_like(idx)
    highest = 0
    for i, v in enumerate(idx):
        out[i] = highest - v
        if out[i] == 0:
            highest += 1
    if np.any(out < 0):
        raise ValueError(
            "indices are not in first-use order, so high-water-mark encoding would wrap. "
            "Run reorder_for_encoding() on the mesh first."
        )
    return out


def reorder_for_encoding(indices: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Permute vertices into first-use order.

    Returns `(perm, remapped_indices)` where `perm[k]` is the original id of the vertex
    that becomes `k`. Real tilers get this ordering for free from vertex-cache
    optimisation; a plain row-major grid mesh does not, because the second vertex of the
    first triangle is a whole row away.
    """
    idx = np.asarray(indices, dtype=np.int64)
    _, first = np.unique(idx, return_index=True)
    perm = idx[np.sort(first)]
    if perm.size != n_vertices:
        # Unreferenced vertices would break the 1:1 mapping; append them at the end.
        used = np.zeros(n_vertices, dtype=bool)
        used[perm] = True
        perm = np.concatenate([perm, np.flatnonzero(~used)])
    inverse = np.empty(n_vertices, dtype=np.int64)
    inverse[perm] = np.arange(n_vertices)
    return perm, inverse[idx]


def decode_indices(encoded: np.ndarray) -> np.ndarray:
    enc = np.asarray(encoded, dtype=np.int64)
    out = np.empty_like(enc)
    highest = 0
    for i, code in enumerate(enc):
        out[i] = highest - code
        if code == 0:
            highest += 1
    return out


# --------------------------------------------------------------------------------------
# Tile
# --------------------------------------------------------------------------------------
@dataclass
class TileBounds:
    """Geodetic extent of one tile, degrees."""

    west: float
    south: float
    east: float
    north: float


@dataclass
class QuantizedMeshTile:
    u: np.ndarray            # uint16, 0..32767 across the tile in longitude
    v: np.ndarray            # uint16, 0..32767 across the tile in latitude
    h: np.ndarray            # uint16, 0..32767 between min_height and max_height
    indices: np.ndarray      # int, triangle vertex indices
    min_height: float
    max_height: float
    centre_ecef: np.ndarray
    sphere_centre: np.ndarray
    sphere_radius: float
    occlusion_point: np.ndarray
    west_indices: np.ndarray
    south_indices: np.ndarray
    east_indices: np.ndarray
    north_indices: np.ndarray
    grid_order: np.ndarray
    """`grid_order[k]` is the row-major grid position of encoded vertex `k`.

    Vertices are stored in first-use order for the index encoding, so this is what maps
    a decoded tile back onto the raster it came from.
    """


def build_tile(heights: np.ndarray, bounds: TileBounds,
               radius_m: float = VENUS_RADIUS_M) -> QuantizedMeshTile:
    """Regular-grid mesh from a `(n, n)` height raster covering `bounds`.

    Row 0 is the north edge, matching the rasters everywhere else in this repo, but
    quantized-mesh `v` increases *northward*, so the rows are flipped exactly once, here.
    """
    grid = np.asarray(heights, dtype=np.float64)
    n_rows, n_cols = grid.shape
    grid = grid[::-1]  # north-up raster -> v increasing north

    lon = np.linspace(bounds.west, bounds.east, n_cols)
    lat = np.linspace(bounds.south, bounds.north, n_rows)
    lon_g, lat_g = np.meshgrid(lon, lat)

    hmin, hmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    span = max(hmax - hmin, 1e-6)

    u = np.rint(np.linspace(0, QUANTIZED_MAX, n_cols)).astype(np.int32)
    v = np.rint(np.linspace(0, QUANTIZED_MAX, n_rows)).astype(np.int32)
    u_g = np.tile(u, n_rows)
    v_g = np.repeat(v, n_cols)
    h_q = np.rint((grid.ravel() - hmin) / span * QUANTIZED_MAX).astype(np.int32)

    ecef = geodetic_to_ecef(lon_g.ravel(), lat_g.ravel(), grid.ravel(), radius_m)
    sphere_centre, sphere_radius = bounding_sphere(ecef)
    centre = geodetic_to_ecef(
        (bounds.west + bounds.east) / 2.0, (bounds.south + bounds.north) / 2.0,
        (hmin + hmax) / 2.0, radius_m,
    )
    occlusion = horizon_occlusion_point(ecef, centre, radius_m)

    # Two triangles per cell. After the north-up flip, row index increases north and
    # column index increases east, so (a, b, c) with b east of a and c north of a winds
    # counter-clockwise seen from outside. Get this backwards and the terrain renders
    # black: every face is culled.
    r, c = np.meshgrid(np.arange(n_rows - 1), np.arange(n_cols - 1), indexing="ij")
    sw = (r * n_cols + c).ravel()
    se, nw, ne = sw + 1, sw + n_cols, sw + n_cols + 1
    indices = np.empty(sw.size * 6, dtype=np.int64)
    indices[0::6], indices[1::6], indices[2::6] = sw, se, ne
    indices[3::6], indices[4::6], indices[5::6] = sw, ne, nw

    n_vertices = n_rows * n_cols
    perm, indices = reorder_for_encoding(indices, n_vertices)
    inverse = np.empty(n_vertices, dtype=np.int64)
    inverse[perm] = np.arange(n_vertices)

    flat = np.arange(n_vertices)
    return QuantizedMeshTile(
        u=u_g.astype(np.int32)[perm], v=v_g.astype(np.int32)[perm], h=h_q[perm],
        indices=indices, min_height=hmin, max_height=hmax,
        centre_ecef=centre, sphere_centre=sphere_centre, sphere_radius=sphere_radius,
        occlusion_point=occlusion,
        west_indices=np.sort(inverse[flat[0::n_cols]]),
        south_indices=np.sort(inverse[flat[:n_cols]]),
        east_indices=np.sort(inverse[flat[n_cols - 1 :: n_cols]]),
        north_indices=np.sort(inverse[flat[-n_cols:]]),
        grid_order=perm,
    )


def encode_tile(tile: QuantizedMeshTile) -> bytes:
    """Serialise to the quantized-mesh-1.0 wire format."""
    out = bytearray()
    out += struct.pack("<3d", *tile.centre_ecef)
    out += struct.pack("<2f", tile.min_height, tile.max_height)
    out += struct.pack("<4d", *tile.sphere_centre, tile.sphere_radius)
    out += struct.pack("<3d", *tile.occlusion_point)

    n = tile.u.size
    out += struct.pack("<I", n)
    for arr in (tile.u, tile.v, tile.h):
        out += delta_zigzag(arr).astype("<u2").tobytes()

    # 16-bit indices while the vertex count allows it, as the spec requires.
    wide = n > 65536
    if wide and len(out) % 4:
        out += b"\x00" * (4 - len(out) % 4)
    dtype = "<u4" if wide else "<u2"
    out += struct.pack("<I", tile.indices.size // 3)
    out += encode_indices(tile.indices).astype(dtype).tobytes()

    for edge in (tile.west_indices, tile.south_indices, tile.east_indices, tile.north_indices):
        out += struct.pack("<I", edge.size)
        out += np.asarray(edge).astype(dtype).tobytes()
    return bytes(out)


def decode_tile(blob: bytes) -> dict:
    """Inverse of `encode_tile`, for the round-trip test and for inspecting a tile
    someone else produced."""
    off = 0
    centre = struct.unpack_from("<3d", blob, off); off += 24
    hmin, hmax = struct.unpack_from("<2f", blob, off); off += 8
    sphere = struct.unpack_from("<4d", blob, off); off += 32
    occlusion = struct.unpack_from("<3d", blob, off); off += 24
    (n,) = struct.unpack_from("<I", blob, off); off += 4

    fields = {}
    for name in ("u", "v", "h"):
        raw = np.frombuffer(blob, dtype="<u2", count=n, offset=off); off += 2 * n
        fields[name] = undelta_zigzag(raw)

    wide = n > 65536
    if wide and off % 4:
        off += 4 - off % 4
    dtype = np.dtype("<u4") if wide else np.dtype("<u2")
    (tri_count,) = struct.unpack_from("<I", blob, off); off += 4
    enc = np.frombuffer(blob, dtype=dtype, count=tri_count * 3, offset=off)
    off += dtype.itemsize * tri_count * 3
    indices = decode_indices(enc)

    edges = {}
    for name in ("west", "south", "east", "north"):
        (count,) = struct.unpack_from("<I", blob, off); off += 4
        edges[name] = np.frombuffer(blob, dtype=dtype, count=count, offset=off).astype(np.int64)
        off += dtype.itemsize * count

    return {
        "centre": np.array(centre), "min_height": hmin, "max_height": hmax,
        "sphere_centre": np.array(sphere[:3]), "sphere_radius": sphere[3],
        "occlusion_point": np.array(occlusion), "vertex_count": n,
        "indices": indices, "edges": edges, **fields,
    }


def dequantize_heights(h: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return min_height + np.asarray(h, dtype=np.float64) / QUANTIZED_MAX * (max_height - min_height)


def write_tile(path: Path, tile: QuantizedMeshTile, gzip_output: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = encode_tile(tile)
    if gzip_output:
        blob = gzip.compress(blob)
    path.write_bytes(blob)
    return path


def layer_json(max_zoom: int, description: str = "ISHTAR learned Venus topography") -> dict:
    """`layer.json` for `CesiumTerrainProvider.fromUrl`.

    `projection: "EPSG:4326"` selects the geodetic (2 x 1) tiling scheme, which is what
    the imagery pyramid uses too. The bounds are the whole planet; the client is told the
    body's radius through the `ellipsoid` option, not through this file.
    """
    return {
        "tilejson": "2.1.0",
        "format": "quantized-mesh-1.0",
        "version": "1.0.0",
        "scheme": "tms",
        "projection": "EPSG:4326",
        "description": description,
        "attribution": "ISHTAR / NASA Magellan",
        "bounds": [-180, -90, 180, 90],
        "tiles": ["{z}/{x}/{y}.terrain?v={version}"],
        "available": [
            [{"startX": 0, "startY": 0, "endX": 2**(z + 1) - 1, "endY": 2**z - 1}]
            for z in range(max_zoom + 1)
        ],
    }


def write_layer_json(root: Path, max_zoom: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "layer.json"
    path.write_text(json.dumps(layer_json(max_zoom), indent=2))
    return path


# --------------------------------------------------------------------------------------
# Pyramid driver
# --------------------------------------------------------------------------------------
def sample_dem(dem: np.ndarray, dem_bounds: TileBounds, tile: TileBounds,
               vertices: int = 65) -> np.ndarray | None:
    """Bilinearly sample a north-up DEM onto one tile's vertex grid.

    Returns None when the tile falls outside the DEM, which is the normal case for a
    regional product: the pyramid is sparse and the client treats a missing tile as
    "not available", not as an error.
    """
    if (tile.east <= dem_bounds.west or tile.west >= dem_bounds.east
            or tile.north <= dem_bounds.south or tile.south >= dem_bounds.north):
        return None

    rows, cols = dem.shape
    lon = np.linspace(tile.west, tile.east, vertices)
    lat = np.linspace(tile.north, tile.south, vertices)  # north-up output grid

    fx = (lon - dem_bounds.west) / (dem_bounds.east - dem_bounds.west) * (cols - 1)
    fy = (dem_bounds.north - lat) / (dem_bounds.north - dem_bounds.south) * (rows - 1)
    fx = np.clip(fx, 0, cols - 1)
    fy = np.clip(fy, 0, rows - 1)

    x0 = np.floor(fx).astype(int); x1 = np.minimum(x0 + 1, cols - 1)
    y0 = np.floor(fy).astype(int); y1 = np.minimum(y0 + 1, rows - 1)
    wx = (fx - x0)[None, :]; wy = (fy - y0)[:, None]

    top = dem[np.ix_(y0, x0)] * (1 - wx) + dem[np.ix_(y0, x1)] * wx
    bot = dem[np.ix_(y1, x0)] * (1 - wx) + dem[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def tile_extent(level: int, x: int, y: int) -> TileBounds:
    """Geodetic (2 x 1) tile extent, TMS row order: y increases north."""
    span = 180.0 / 2**level
    west = -180.0 + x * span
    south = -90.0 + y * span
    return TileBounds(west, south, west + span, south + span)


def _overlaps(a: TileBounds, b: TileBounds) -> bool:
    return not (a.east <= b.west or a.west >= b.east or a.north <= b.south or a.south >= b.north)


def tiles_in(level: int, box: TileBounds | None):
    """Tile indices at `level` that intersect `box` (all of them when box is None).

    Iterating the whole level and testing each tile is fine to level 9 (524 288 tiles) but
    not beyond, which is exactly where regions of interest start — so the range is derived
    from the box instead of filtered from the full grid.
    """
    nx, ny = 2 ** (level + 1), 2**level
    if box is None:
        yield from ((x, y) for x in range(nx) for y in range(ny))
        return
    span = 180.0 / 2**level
    x0 = max(0, int(math.floor((box.west + 180.0) / span)))
    x1 = min(nx - 1, int(math.ceil((box.east + 180.0) / span)))
    y0 = max(0, int(math.floor((box.south + 90.0) / span)))
    y1 = min(ny - 1, int(math.ceil((box.north + 90.0) / span)))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            yield x, y


def build_pyramid(
    dem: np.ndarray,
    dem_bounds: TileBounds,
    root: Path,
    max_level: int = 9,
    min_level: int = 0,
    vertices: int = 65,
    radius_m: float = VENUS_RADIUS_M,
    box: TileBounds | None = None,
) -> dict[str, int]:
    """Write a `{z}/{x}/{y}.terrain` tree plus `layer.json`.

    Tiles are gzipped, which the format expects and which halves the pyramid: a level-9
    Venus tile is ~73 kB raw and ~9 kB compressed.

    `box` restricts the levels to a region of interest, which is how the real product
    ships — global coverage to level 9, and level 12 only inside a handful of boxes,
    because level 12 globally would be 33 million tiles.
    """
    written = {}
    for level in range(min_level, max_level + 1):
        count = 0
        for x, y in tiles_in(level, box):
            bounds = tile_extent(level, x, y)
            if box is not None and not _overlaps(bounds, box):
                continue
            heights = sample_dem(dem, dem_bounds, bounds, vertices)
            if heights is None:
                continue
            tile = build_tile(heights, bounds, radius_m)
            write_tile(root / str(level) / str(x) / f"{y}.terrain", tile)
            count += 1
        written[str(level)] = count
    write_layer_json(root, max_level)
    return written
