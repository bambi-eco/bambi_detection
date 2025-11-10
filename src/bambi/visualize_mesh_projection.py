import sys
import numpy as np
import trimesh
from alfspy.render.render import read_gltf
from trimesh import Trimesh
from trimesh import transformations as tf

try:
    import pyrender
    HAS_PYRENDER = True
except Exception:
    HAS_PYRENDER = False

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
except Exception:
    from trimesh.ray.ray_triangle import RayMeshIntersector

import numpy as np
import svgwrite

def project_points(points, V, P, W=1280, H=720):
    """Project Nx3 points with view (V) + projection (P) to pixel coords (W,H)."""
    N = points.shape[0]
    homo = np.hstack([points, np.ones((N, 1))])            # (N,4)
    clip = (P @ (V @ homo.T)).T                            # (N,4)
    ndc = clip[:, :3] / np.clip(clip[:, 3:4], 1e-9, np.inf)  # perspective divide
    # NDC [-1..1] -> pixels; flip Y for SVG (origin at top-left)
    x = (ndc[:, 0] * 0.5 + 0.5) * W
    y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    return np.vstack([x, y]).T, ndc[:, 2]  # return (x,y) and NDC z

def draw_point_circle(dwg, point_world, V, P, W, H, radius_px=8, color="black"):
    # homogeneous
    pw = np.array(point_world, dtype=float).reshape(1,3)
    pts2d, _ = project_points(pw, V, P, W, H)
    x, y = float(pts2d[0,0]), float(pts2d[0,1])

    # draw circle
    dwg.add(dwg.circle(center=(x, y),
                       r=radius_px,
                       fill=color,
                       stroke=color,
                       stroke_width=1))

def export_scene_svg(mesh, face_mask_inside, frustum_lines, V, P,
                     trimmed_edges=None,  # list of (q0,q1) 3D; optional
                     point_above=None,
                     W=1280, H=720, out_path="scene.svg"):
    """
    - mesh: trimesh.Trimesh
    - face_mask_inside: boolean mask over faces (red fill)
    - frustum_lines: list of (p0,p1) 3D pairs (untrimmed)
    - trimmed_edges: list of (q0,q1) 3D pairs (trimmed by rays); if given, use these
    """
    dwg = svgwrite.Drawing(out_path, profile='tiny',
                           size=(f"{W}px", f"{H}px"),
                           viewBox=f"0 0 {W} {H}")

    # ---- draw mesh triangles (light gray), and fill red for inside-frustum faces
    Verts = mesh.vertices
    F = mesh.faces
    # (optional) sort by depth so farther triangles are drawn first (simple painter’s algo)
    tri_centers = mesh.triangles_center
    _, zc = project_points(tri_centers, V, P, W, H)
    order = np.argsort(zc)  # far to near
    for fi in order:
        tri = Verts[F[fi]]
        pts2d, _ = project_points(tri, V, P, W, H)
        pts = [(float(pts2d[k,0]), float(pts2d[k,1])) for k in range(3)]
        inside = bool(face_mask_inside[fi]) if face_mask_inside is not None else False
        fill = svgwrite.rgb(255, 0, 0) if inside else "none"
        fill_opacity = 0.6 if inside else 0.0
        stroke = svgwrite.rgb(200, 200, 200)
        dwg.add(dwg.polygon(points=pts, fill=fill, fill_opacity=fill_opacity,
                            stroke=stroke, stroke_width=0.6))

    if point_above is not None:
        draw_point_circle(dwg, point_above, V, P, W, H,
                          radius_px=8,
                          color="black")

    # ---- draw frustum edges (cyan)
    edge_color = svgwrite.rgb(0, 0, 0)
    edges = trimmed_edges if trimmed_edges is not None else frustum_lines
    for p0, p1 in edges:
        pts2d, _ = project_points(np.vstack([p0, p1]), V, P, W, H)
        x0,y0 = map(float, pts2d[0]); x1,y1 = map(float, pts2d[1])
        dwg.add(dwg.line(start=(x0,y0), end=(x1,y1), stroke=edge_color, stroke_width=5.0))

    dwg.save()
    print(f"SVG written to {out_path}")

def mesh_center(mesh: trimesh.Trimesh) -> np.ndarray:
    # center of the mesh's axis-aligned bounding box
    bounds = mesh.bounds  # shape (2, 3): min, max
    return bounds.mean(axis=0)


def extract_submesh_aabb(mesh: trimesh.Trimesh, center: np.ndarray,
                         size_xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    """Return a submesh containing faces whose ALL vertices lie inside the AABB."""
    sx, sy, sz = size_xyz
    mins = center - 0.5 * np.array([sx, sy, sz])
    maxs = center + 0.5 * np.array([sx, sy, sz])

    verts = mesh.vertices
    faces = mesh.faces

    # For each face, check if all 3 vertices are inside the AABB
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

    def inside(v):
        return np.all(v >= mins, axis=1) & np.all(v <= maxs, axis=1)

    mask = inside(v0) & inside(v1) & inside(v2)
    if not np.any(mask):
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64), process=False)

    sub_faces = faces[mask]
    submesh = mesh.submesh([mask], append=True, repair=False)
    return submesh


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Create a right-handed view matrix (4x4) from eye->target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.eye(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4)
    T[:3, 3] = -eye
    return M @ T


def perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    """OpenGL-style perspective projection matrix (4x4)."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    P = np.zeros((4, 4))
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (z_far + z_near) / (z_near - z_far)
    P[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
    P[3, 2] = -1.0
    return P


def frustum_planes_from_matrices(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Extract 6 view-frustum planes (nx, ny, nz, d) from combined clip matrix C = P @ V.
    Each plane eq: n.x * X + n.y * Y + n.z * Z + d >= 0 means 'inside'.
    """
    C = P @ V
    planes = []
    # Left:  C[3] + C[0]
    planes.append(C[3, :] + C[0, :])
    # Right: C[3] - C[0]
    planes.append(C[3, :] - C[0, :])
    # Bottom: C[3] + C[1]
    planes.append(C[3, :] + C[1, :])
    # Top:    C[3] - C[1]
    planes.append(C[3, :] - C[1, :])
    # Near:   C[3] + C[2]
    planes.append(C[3, :] + C[2, :])
    # Far:    C[3] - C[2]
    planes.append(C[3, :] - C[2, :])

    planes = np.array(planes)
    # normalize plane normals
    for i in range(6):
        n = planes[i, :3]
        norm = np.linalg.norm(n)
        if norm > 0:
            planes[i, :] /= norm
    return planes  # shape (6, 4)


def points_inside_frustum(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """
    Test Nx3 points against 6 planes (nx, ny, nz, d).
    Returns boolean mask: True if point lies inside all planes.
    """
    # For each plane, compute dot(n, p) + d
    # points: (N,3), planes: (6,4)
    N = points.shape[0]
    p_h = np.hstack([points, np.ones((N, 1))])  # (N,4)
    dists = p_h @ planes.T  # (N,6)
    return np.all(dists >= 0, axis=1)


def cylinder_between(p0, p1, radius, sections=16):
    v = p1 - p0
    h = np.linalg.norm(v)
    if h < 1e-9:
        return None

    # Cylinder centered at origin, along +Z, height=h
    cyl = trimesh.creation.cylinder(radius=radius, height=h, sections=sections)

    # Rotate +Z to direction v
    z = np.array([0.0, 0.0, 1.0])
    dir = v / h
    axis = np.cross(z, dir)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-12:
        # Already aligned or opposite
        angle = 0.0 if np.dot(z, dir) > 0 else np.pi
        R = tf.rotation_matrix(angle, [1.0, 0.0, 0.0])  # any axis
    else:
        angle = np.arccos(np.clip(np.dot(z, dir), -1.0, 1.0))
        R = tf.rotation_matrix(angle, axis / axis_len)

    # Move cylinder so its base is at p0 (default cylinder is centered at origin)
    # After rotation, cylinder runs from z=-h/2 to z=+h/2. Shift by +h/2 along its local +Z,
    # then translate to p0.
    T_up = tf.translation_matrix([0, 0, h * 0.5])
    T_to_p0 = tf.translation_matrix(p0)

    M = T_to_p0 @ R @ T_up
    cyl.apply_transform(M)
    return cyl

def select_faces_inside_frustum(mesh: trimesh.Trimesh, planes: np.ndarray) -> np.ndarray:
    """
    Fast selection: mark faces whose centroids are inside the frustum.
    Returns boolean mask over faces.
    """
    tri_centroids = mesh.triangles_center  # (F,3)
    return points_inside_frustum(tri_centroids, planes)


def colorize_faces(mesh: trimesh.Trimesh, face_mask: np.ndarray, color_inside=(255, 0, 0, 255),
                   color_outside=(200, 200, 200, 255)) -> None:
    """Assigns per-face colors (in-place)."""
    if mesh.visual.face_colors is None or len(mesh.visual.face_colors) != len(mesh.faces):
        mesh.visual.face_colors = np.tile(color_outside, (len(mesh.faces), 1))
    mesh.visual.face_colors[:] = color_outside
    mesh.visual.face_colors[face_mask] = color_inside

# 2) Robust segment trimming using ALL hits on p0 -> p1
def shorten_to_mesh_robust(ray_intersector, p0, p1, tol=1e-8):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < tol:
        return p0, p1  # degenerate edge

    d = v / L  # normalize

    # Collect all intersections along the ray from p0 in direction d
    locs, idx_ray, idx_tri = ray_intersector.intersects_location(
        p0.reshape(1, 3), d.reshape(1, 3)
    )
    if len(locs) == 0:
        # No forward hits; try from the other end backward
        locs_b, _, _ = ray_intersector.intersects_location(
            p1.reshape(1, 3), (-d).reshape(1, 3)
        )
        if len(locs_b) == 0:
            # No hits either way -> keep original edge (or skip, your choice)
            return p0, p1

        # Distances from p1 backward, clamp to the segment length
        t_back = np.dot((locs_b - p1), -d)
        t_back = t_back[np.isfinite(t_back)]
        t_back = t_back[(t_back >= -tol) & (t_back <= L + tol)]
        if len(t_back) == 0:
            return p0, p1

        tmin = float(np.min(np.clip(t_back, 0.0, L)))
        # Trim from p1 backward
        return p1 - d * tmin, p1

    # Distances from p0 forward, keep those ON the segment
    t = np.dot((locs - p0), d)
    t = t[np.isfinite(t)]
    t = t[(t >= -tol) & (t <= L + tol)]
    if len(t) == 0:
        return p0, p1

    tmin = float(np.min(np.clip(t, 0.0, L)))
    # Trim p1 to the first hit
    return p0, p0 + d * tmin

def main():
    mesh_data, texture_data = read_gltf(r"Z:\correction_data\0_dem.glb")
    mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    if not isinstance(mesh, trimesh.Trimesh):
        # If it’s a Scene, take the first geometry merged
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
        else:
            raise ValueError("Loaded object is not a mesh/scene.")


    # 1) Center point of mesh
    center = mesh_center(mesh)
    print("Mesh AABB center:", center)

    # 2) Extract submesh in a defined AABB (edit sizes as needed)
    size_xyz = (500.0, 500.0, 180.0)
    submesh = extract_submesh_aabb(mesh, center, size_xyz)
    # submesh = mesh
    mins, maxs = submesh.bounds
    extent = maxs - mins
    print(f"Submesh faces: {len(submesh.faces)} (AABB size={extent})")

    ray = RayMeshIntersector(submesh)


    # 3) A point 10 units along +Z from the center
    point_above = center + np.array([0.0, 0.0, 10.0])
    print("Point +10 on Z from center:", point_above)

    # 4) Build a camera frustum from this point looking toward the mesh center
    eye = point_above
    target = center
    up = np.array([0.0, 1.0, 0.0])
    V = look_at(eye, target, up)

    # Frustum params (edit as you like)
    fovy_deg = 60.0
    aspect = 16.0 / 9.0
    z_near = 1.0
    z_far = 1000.0
    P = perspective(fovy_deg, aspect, z_near, z_far)

    planes = frustum_planes_from_matrices(V, P)

    # 5) Find faces inside the frustum (by centroid) and color them red
    mask_inside = select_faces_inside_frustum(submesh, planes)
    colorize_faces(submesh, mask_inside, color_inside=(255, 0, 0, 255), color_outside=(180, 180, 180, 255))

    # Also color the AABB submesh differently to show the crop (optional)
    # if len(submesh.faces) > 0:
    #     submesh.visual.face_colors = np.tile([0, 255, 0, 140], (len(submesh.faces), 1))  # translucent green

    # 6) Visualize
    # Option A: quick viewer via trimesh
    scene = trimesh.Scene()
    # scene.add_geometry(mesh)
    if len(submesh.faces) > 0:
        scene.add_geometry(submesh)

    # Add tiny spheres for the two points: center and point_above
    radius = submesh.scale * 0.005 if submesh.scale > 0 else 1.0
    sphere_center = trimesh.creation.icosphere(radius=radius)
    sphere_center.apply_translation(center)
    # sphere_center.visual.face_colors = [0, 0, 255, 255]
    # scene.add_geometry(sphere_center)

    sphere_above = sphere_center.copy()
    sphere_above.apply_translation(point_above - center)
    sphere_above.visual.face_colors = [0, 0, 0, 255]
    scene.add_geometry(sphere_above)

    # Optional: draw a simple frustum wireframe for reference
    def frustum_corners_world(eye, target, up, fovy, aspect, z_near, z_far):
        V = look_at(eye, target, up)
        # Build camera basis
        # Inverse view to get camera axes in world:
        Vinv = np.linalg.inv(V)
        cam_pos = Vinv[:3, 3]
        cam_x = Vinv[:3, 0]
        cam_y = Vinv[:3, 1]
        cam_z = -Vinv[:3, 2]  # forward

        def plane_corners(z):
            h = 2 * z * np.tan(np.radians(fovy) / 2)
            w = h * aspect
            c = cam_pos + cam_z * z
            dx = cam_x * (w / 2)
            dy = cam_y * (h / 2)
            return np.array([
                c - dx - dy,
                c + dx - dy,
                c + dx + dy,
                c - dx + dy
            ])

        n4 = plane_corners(z_near)
        f4 = plane_corners(z_far)
        return n4, f4

    n4, f4 = frustum_corners_world(eye, target, up, fovy_deg, aspect, z_near, z_far)
    frustum_pts = np.vstack([n4, f4])
    # Lines between corners
    # lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = [(0, 4), (1, 5), (2, 6), (3, 7)]

    # pick a visible radius based on model size
    mins, maxs = submesh.bounds
    diag = float(np.linalg.norm(maxs - mins)) or 1.0
    edge_radius = diag * 0.002  # adjust if too thin/thick
    trimmed_edges = []

    for i, j in lines:
        p0, p1 = frustum_pts[i], frustum_pts[j]
        q0, q1 = shorten_to_mesh_robust(ray, p0, p1)

        if np.linalg.norm(q1 - q0) < 1e-6:
            continue

        tube = cylinder_between(q0, q1, radius=edge_radius, sections=20)
        if tube is None:
            continue
        tube.visual.face_colors = [0, 0, 0, 255]
        scene.add_geometry(tube)
        trimmed_edges.append((q0, q1))

    W, H = 1920, 1080
    fov_y_deg = 60.0  # initial; you can change it in the viewer with scroll/zoom
    cam = trimesh.scene.cameras.Camera(
        resolution=(W, H),
        fov=(None, fov_y_deg),  # (fov_x, fov_y) in degrees; None lets trimesh infer fov_x from aspect
        z_near=0.1,
        z_far=10000
    )
    scene.camera = cam
    scene.camera_transform = np.eye(4)  # start pose

    # 3) Launch the interactive viewer. This blocks until you close the window.
    scene.show()  # close when you’re happy with the view

    # 4) After the window closes, grab the final camera and export SVG
    cam_tf = scene.camera_transform.copy()  # world-from-camera
    W, H = scene.camera.resolution
    fov_x_deg, fov_y_deg = scene.camera.fov

    # Build view + projection from the viewer’s camera
    V = np.linalg.inv(cam_tf)
    aspect = W / float(H)

    def perspective_from_fov(fovy_deg, aspect, z_near, z_far):
        f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
        P = np.zeros((4, 4))
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (z_far + z_near) / (z_near - z_far)
        P[2, 3] = (2 * z_far * z_near) / (z_near - z_far)
        P[3, 2] = -1.0
        return P

    # choose generous near/far from mesh size
    diag = float(np.linalg.norm(maxs - mins)) or 1.0
    P = perspective_from_fov(fovy_deg=fov_y_deg, aspect=aspect,
                             z_near=diag * 1e-3, z_far=diag * 10.0)

    target_faces = 50000
    mesh_simplified = submesh.simplify_quadratic_decimation(target_faces)
    mask_inside_simpl = select_faces_inside_frustum(mesh_simplified, planes)
    export_scene_svg(
        mesh=mesh_simplified,
        face_mask_inside=mask_inside_simpl,  # or None if you just want outlines
        frustum_lines=[],  # unused if trimmed_edges provided
        trimmed_edges=trimmed_edges,  # draw your cyan tubes in 2D as lines
        V=V, P=P,
        W=W, H=H,
        point_above=point_above,
        out_path="scene.svg"
    )
    print("SVG written to scene.svg using the final interactive camera.")

    # if HAS_PYRENDER:
    #     # prettier viewer (lights + orbit controls)
    #     trimesh_meshes = [
    #         g for g in scene.geometry.values()
    #         if isinstance(g, trimesh.Trimesh)
    #     ]
    #
    #     render_scene = pyrender.Scene()
    #     for m in trimesh_meshes:
    #         render_scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    #
    #     # add camera for visualization (not used for culling; just a viewer camera)
    #     cam = pyrender.PerspectiveCamera(yfov=np.radians(fovy_deg), aspectRatio=aspect)
    #     cam_node = pyrender.Node(camera=cam, matrix=np.linalg.inv(look_at(eye, target, up)))
    #     render_scene.add_node(cam_node)
    #
    #     # lights
    #     render_scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
    #     pyrender.Viewer(render_scene, use_raymond_lighting=True)
    # else:
    #     scene.show()


if __name__ == "__main__":
    main()
