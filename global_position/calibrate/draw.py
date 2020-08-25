import numpy as np
from shapely.geometry import LineString, Polygon
import cv2

W, H = 28.65, 15.24


def make_field_circle(center, r, nn=1):
    """
    Returns points that lie on a circle on the ground
    :param r: radius
    :param nn: points per arc?
    :return: 3D points on a circle with y = 0
    """
    cx, cy, cz = center[0], center[1], center[2]
    d = 2 * np.pi * r
    n = int(nn * d)
    return [(cx+np.cos(2 * np.pi / n * x) * r, cy+0, cz+np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


def get_field_points():
    # rectangles
    outer_rect = np.array([[-H/2., 0, -W/2.],
                            [-H/2., 0, W/2.],
                            [H/2., 0, W/2.],
                            [H/2., 0, -W/2.]])
    l_ft_rect = np.array([[-2.45, 0., -W/2.],
                         [ 2.45, 0., -W/2.],
                         [ 2.45, 0., -W/2.+5.79],
                         [-2.45, 0., -W/2.+5.79]])
    r_ft_rect = np.array([[-2.45, 0., W/2.-5.79],
                         [ 2.45, 0., W/2.-5.79],
                         [ 2.45, 0., W/2.],
                         [-2.45, 0., W/2.]])
    # lines
    mid_line = np.array([[H/2., 0.,0.],
                        [-H/2., 0., 0.]])
    ul_3pt_line = np.array([[H/2.-0.91, 0.,-W/2.],
                                [H/2.-0.91, 0., -W/2.+4.26]])
    bl_3pt_line = np.array([[-H/2.+0.91, 0.,-W/2.],
                                [-H/2.+0.91, 0., -W/2.+4.26]])
    ur_3pt_line = np.array([[H/2.-0.91, 0.,W/2.],
                                [H/2.-0.91, 0., W/2.-4.26]])
    br_3pt_line = np.array([[-H/2.+0.91, 0.,W/2.],
                                [-H/2.+0.91, 0., W/2.-4.26]])

    # circles
    central_circle = np.array(make_field_circle(center=(0,0,0), r=1.83, nn=7))

    l_ft_circle = np.array(make_field_circle(center=(0,0,-W/2.+5.79), r=1.83, nn=7))
    index = l_ft_circle[:, 2] > (-W/2.+5.79)
    l_ft_circle = l_ft_circle[index, :]

    r_ft_circle = np.array(make_field_circle(center=(0,0,W/2.-5.79), r=1.83, nn=7))
    index = r_ft_circle[:, 2] < (W/2.-5.79)
    r_ft_circle = r_ft_circle[index, :]

    l_restricted_circle = np.array(make_field_circle(center=(0,0,-W/2.+1.575), r=1.21, nn=7))
    index = l_restricted_circle[:, 2] > (-W/2.+1.575)
    l_restricted_circle = l_restricted_circle[index, :]

    r_restricted_circle = np.array(make_field_circle(center=(0,0,W/2.-1.575), r=1.21, nn=7))
    index = r_restricted_circle[:, 2] < (W/2.-1.575)
    r_restricted_circle = r_restricted_circle[index, :]

    l_3pt_circle = np.array(make_field_circle(center=(0,0,-W/2.+1.575), r=7.24))
    index = l_3pt_circle[:, 2] > (-W/2.+4.26)
    l_3pt_circle = l_3pt_circle[index, :]

    r_3pt_circle = np.array(make_field_circle(center=(0,0,W/2.-1.575), r=7.24))
    index = r_3pt_circle[:, 2] < (W/2.-4.26)
    r_3pt_circle = r_3pt_circle[index, :]

    return [outer_rect, l_ft_rect, r_ft_rect,
            mid_line, ul_3pt_line, bl_3pt_line,
            ur_3pt_line, br_3pt_line,
            central_circle, l_ft_circle, r_ft_circle,
            l_restricted_circle, r_restricted_circle,
            l_3pt_circle, r_3pt_circle]


def project_field_to_image(camera):

    field_list = get_field_points()

    field_points2d = []
    for i in range(len(field_list)):
        tmp, depth = camera.project(field_list[i])

        behind_points = (depth < 0).nonzero()[0]
        tmp[behind_points, :] *= -1

        field_points2d.append(tmp)

    return field_points2d


def draw_field(camera):

    field_points2d = project_field_to_image(camera)
    h, w = camera.height, camera.width
    # Check if the entities are 15
    assert len(field_points2d) == 15

    img_polygon = Polygon([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the boxes
    for i in range(3):

        # And make a new image with the projected field
        linea = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        lineb = LineString([(field_points2d[i][1, :]),
                            (field_points2d[i][2, :])])

        linec = LineString([(field_points2d[i][2, :]),
                            (field_points2d[i][3, :])])

        lined = LineString([(field_points2d[i][3, :]),
                            (field_points2d[i][0, :])])

        if i == 0:
            polygon0 = Polygon([(field_points2d[i][0, :]),
                                (field_points2d[i][1, :]),
                                (field_points2d[i][2, :]),
                                (field_points2d[i][3, :])])

            intersect0 = img_polygon.intersection(polygon0)
            if not intersect0.is_empty:
                pts = np.array(list(intersect0.exterior.coords), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(mask, pts, (255, 255, 255))

        intersect0 = img_polygon.intersection(linea)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lineb)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(linec)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

        intersect0 = img_polygon.intersection(lined)
        if not intersect0.is_empty:
            pts = np.array(list(list(intersect0.coords)), dtype=np.int32)
            if pts.shape[0] < 2:
                continue
            cv2.line(canvas, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), (255, 255, 255))

    # lines
    for i in range(3, 8):
        line1 = LineString([(field_points2d[i][0, :]),
                            (field_points2d[i][1, :])])

        intersect1 = img_polygon.intersection(line1)
        if not intersect1.is_empty:
            pts = np.array(list(list(intersect1.coords)), dtype=np.int32)
            pts = pts[:, :].reshape((-1, 1, 2))
            cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    # Circles
    for ii in range(8, 15):
        for i in range(field_points2d[ii].shape[0] - 1):
            line2 = LineString([(field_points2d[ii][i, :]),
                                (field_points2d[ii][i + 1, :])])
            intersect2 = img_polygon.intersection(line2)
            if not intersect2.is_empty:
                pts = np.array(list(list(intersect2.coords)), dtype=np.int32)
                pts = pts[:, :].reshape((-1, 1, 2))
                cv2.fillConvexPoly(canvas, pts, (255, 255, 255), )

    return canvas[:, :, 0] / 255., mask[:, :, 0] / 255.
