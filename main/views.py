import base64
import io
import json
import math
import random
from shapely.geometry import LineString, MultiLineString,Polygon, Point

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as Colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from django.views.decorators.csrf import csrf_exempt

from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import View


class HomeView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'main/index.html')


def f_3f(n):
    # format float number
    return "%.3f" % n


def f_1f(n):
    # format float number
    return "%.1f" % n


class LeastSquaresView(View):

    @staticmethod
    def get_results_context(points):
        points = np.array(points)
        x, y = points.T
        print('------\n x', x, 'y', y)

        ptp = np.matmul(points.T, points)
        m = len(points)

        # Geo center, c = sum(p).1/m : m is len(p)

        c = np.matrix([sum(x), sum(y)]) / m
        print('------\nc: \n', c)

        ctc = np.matmul(c.transpose(), c)

        # Matrix M: M = m.cT.c - pT.p

        M = m * ctc - ptp

        print('------\nM: \n', M)

        # Calculate eigenvals, eigenvecs: M.[x y] = eigen[x y]
        # Notice: The eigenvectors returned by the numpy.linalg.eig() function are normalized.
        eigenvals, eigenvecs = np.linalg.eig(M)

        # tmp_eigenvecs = la.eigh(M)
        # eigenvecs = eigenvecs.T

        print('------\n Eigenvals:\n', eigenvals)
        print('------\n Eigenvecs:\n', eigenvecs)

        eigenval_max = max(eigenvals)
        print('------\n eigenval_max:', eigenval_max)

        # we need only the max eigenval
        eigenval_max_position = int(np.where(eigenvals == eigenval_max)[0][0])
        print('------\n position:', eigenval_max_position)

        # get vector by position
        # eigenvectors of eig are in the columns, not the rows.
        # eigenvec = eigenvecs.T[eigenval_max_position]
        eigenvec = eigenvecs[:, eigenval_max_position]  # return col with eigenval_max_position

        print('------\n eigenvec:\n', eigenvec)

        # check, if correct return True
        # print(np.allclose(np.dot(M,eigenvecs[:,0]), np.dot(eigenvals[0],eigenvecs[:,0])))

        a, b = eigenvec.item(0), eigenvec.item(1)

        # transform it so we can eigenvec * c
        eigenvec = np.array([
            [a, b]
        ])

        print('------\n transformed eigenvec:\n ', eigenvec)

        d = (- eigenvec * c.T).item(0)
        print('------\n D:', d)

        # Figure init

        fig = plt.figure()

        ax = plt.gca()

        ax.scatter(x, y)

        # move x, y axes to center

        ax.grid(True)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        # optimality line

        _ = plt.plot(x, y, 'o', label='Original data', markersize=5)
        _ = plt.plot(c.item(0), c.item(1), 'o', label='Geo center', markersize=5)

        label = f'{f_3f(a)}x + {f_3f(b)}y +({f_3f(d)})'

        if a and b:
            formula = -(a * x + d) / b  # cause ax + by + d = 0
            _ = plt.plot(x, formula, 'r', label=label)

        elif b and not a:
            formula = - d / b
            _ = plt.hlines(formula, min(x), max(x), 'r', label=label)

        elif a and not b:
            formula = - d / a
            _ = plt.vlines(formula, min(y), max(y), 'r', label=label)

        _ = plt.legend()

        # grid devisions

        plt.xlim(-max(x) - 1, max(x) + 1)
        plt.ylim(-max(y) - 1, max(y) + 1)

        # Create png
        canvas = FigureCanvas(fig)
        outstr = io.BytesIO()
        canvas.print_png(outstr)
        inline_png = base64.b64encode(outstr.getvalue()).decode('ascii')
        outstr.close()

        ctx=dict(
            c_value=[f_3f(c.item(0)), f_3f(c.item(1))],
            inline_png=inline_png,
            m_matrix=M,
            eigen_vals=eigenvals,
            eigen_vecs=eigenvecs.T
        )

        return ctx

    @staticmethod
    def fetch_values(form_data_list, name):
        return [int(item['value']) for item in form_data_list if item['name'].startswith(f'{name}')]

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        form_data = request.POST.get('formData')
        form_data_list = json.loads(form_data)

        # filter elements keep needed ones only
        form_data_list = [item for item in form_data_list if item['name'].startswith(('x_', 'y_'))]
        # fetch out x & y
        x_vals, y_vals = self.fetch_values(form_data_list, 'x_'), self.fetch_values(form_data_list, 'y_')
        x_y_vals = [list(item) for item in list(zip(x_vals, y_vals))]

        ctx = dict()
        ctx['x_vals'] = x_vals
        ctx['y_vals'] = y_vals
        ctx.update(**self.get_results_context(x_y_vals))

        return render(request, 'main/least_squares/includes/results_content.html', ctx)

    def get(self, request, *args, **kwargs):

        return render(request, 'main/least_squares/index.html')


class TangentTwoCirclesView(View):

    @staticmethod
    def get_results_context(c1_data, c2_data):
        x1, y1, r1 = c1_data[0], c1_data[1], c1_data[2],
        x2, y2, r2 = c2_data[0], c2_data[1], c2_data[2],

        x = np.array([x1, x2])
        y = np.array([y1, y2])
        # TODO: r should be > 0
        r = np.array([r1, r2])
        ctx = dict()

        # Distance between circles, NOT centers
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) - (r2 + r1)

        # distance between centers
        distance_centers = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        print('\n ------ Distance: \n', distance)

        # make the size proportional to the distance from the origin
        s = [0.1 * np.linalg.norm([a, b]) for a, b in zip(x, y)]
        s = [a / max(s) for a in s]  # scale

        # set color based on size
        c = s
        colors = [cm.jet(color) for color in c]  # gets the RGBA values from a float

        # create a new figure
        fig = plt.figure()
        ax = plt.gca()

        for a, b, color, radius in zip(x, y, colors, r):
            # plot circles using the RGBA colors
            circle = plt.Circle((a, b), radius, color=color, fill=False)
            ax.add_artist(circle)
            ax.add_patch(circle)
            # ax.add_patch(patches.Circle((a, b), radius, color=color, alpha=0.1, label='laaabel'))

        # TODO: Special case: two equal radii circles

        if distance >= 1:

            """
            Keep r0 always the bigger value
            (x - a)^2 + (y - b)^2 = ro^2
            (x - c)^2 + (y - d)^2 = roo^2
            """
            # if r1 > r2:
            #     a, b, c, d, ro, roo = x1, y1, x2, y2, r1, r2
            # else:
            #     a, b, c, d, ro, roo = x2, y2, x1, y1, r2, r1
            a, b, c, d, ro, roo = x2, y2, x1, y1, r2, r1
            # Intersection of inner tangent lines
            xo = (c * ro + a * roo) / (ro + roo)
            yo = (d * ro + b * r1) / (ro + roo)

            xt1 = ((ro ** 2 * (xo - a) + ro * (yo - b) * math.sqrt(
                (xo - a) ** 2 + (yo - b) ** 2 - ro ** 2)) / ((xo - a) ** 2 + (yo - b) ** 2)) + a

            xt2 = ((ro ** 2 * (xo - a) - ro * (yo - b) * math.sqrt(
                (xo - a) ** 2 + (yo - b) ** 2 - ro ** 2)) / ((xo - a) ** 2 + (yo - b) ** 2)) + a

            yt1 = ((ro ** 2 * (yo - b) + ro * (xo - a) * math.sqrt(
                (xo - a) ** 2 + (yo - b) ** 2 - ro ** 2)) / ((xo - a) ** 2 + (yo - b) ** 2)) + b

            yt2 = ((ro ** 2 * (yo - b) - ro * (xo - a) * math.sqrt(
                (xo - a) ** 2 + (yo - b) ** 2 - ro ** 2)) / ((xo - a) ** 2 + (yo - b) ** 2)) + b

            # Check,
            correctness_check = bool((b - yt1) * (yo - yt1) / (xt1 - a) * (xt1 - xo) == 1)

            # if s != 1 the point is not correct, swap the values Yt1 with Yt2
            if not correctness_check:
                yt1, yt2 = yt2, yt1

            xt3 = ((roo ** 2 * (xo - c) + roo * (yo - d) * math.sqrt(
                (xo - c) ** 2 + (yo - d) ** 2 - roo ** 2)) / ((xo - c) ** 2 + (yo - d) ** 2)) + c

            xt4 = ((roo ** 2 * (xo - c) - roo * (yo - d) * math.sqrt(
                (xo - c) ** 2 + (yo - d) ** 2 - roo ** 2)) / ((xo - c) ** 2 + (yo - d) ** 2)) + c

            yt3 = ((roo ** 2 * (yo - d) + roo * (xo - c) * math.sqrt(
                (xo - c) ** 2 + (yo - d) ** 2 - roo ** 2)) / ((xo - c) ** 2 + (yo - d) ** 2)) + d

            yt4 = ((roo ** 2 * (yo - d) - roo * (xo - c) * math.sqrt(
                (xo - c) ** 2 + (yo - d) ** 2 - roo ** 2)) / ((xo - c) ** 2 + (yo - d) ** 2)) + d

            # Check
            correctness_check = bool((d - yt3) * (yo - yt3) / (xt3 - xo) * (xt3 - xo) == 1)

            if not correctness_check:
                yt3, yt4 = yt4, yt3

            # Lines equations

            formula_1 = (x - xt1) * (yo - yt1) / (xo - xt1) + yt1
            formula_2 = (x - xt2) * (yo - yt2) / (xo - xt2) + yt2

            # _ = plt.plot(x, formula_1, 'm', label='Y1')
            # _ = plt.plot([xt1, xt3], [yt1, yt3], label='Y1')
            # _ = plt.plot([xt2, xt4], [yt2, yt4], label='Y2')

            # _ = plt.plot(x, formula_2, 'c', label='Y2')
            # let us try using Shapely
            line1 = LineString([(xt1, yt1), (xt3, yt3)])
            line2 = LineString([(xt2, yt2), (xt4, yt4)])

            xl1, yl1 = line1.xy
            ax.plot(xl1, yl1)

            xl2, yl2 = line2.xy
            ax.plot(xl2, yl2)

            # Intersection point
            xl12, yl12 = line1.intersection(line2).xy

            # show the point
            # _ = plt.plot(xo, yo, 'o', label=f'XO ({f_1f(xo)}, {f_1f((yo))})', markersize=5)
            _ = plt.plot(xl12, yl12, 'o', label=f'XO ({f_1f(xo)}, {f_1f((yo))})', markersize=5)
            # _ = plt.plot(27.7, 13.0, 'o', label=f'XO ({f_1f(xo)}, {f_1f((yo))})', markersize=5)
            _ = plt.plot(xt1, yt1, 'o', label=f'XYt1 ({f_1f(xt1)}, {f_1f(yt1)})', markersize=5)
            _ = plt.plot(xt2, yt2, 'o', label=f'XYt2 ({f_1f(xt2)}, {f_1f(yt2)})', markersize=5)
            _ = plt.plot(xt3, yt3, 'o', label=f'XYt3 ({f_1f(xt3)}, {f_1f(yt3)})', markersize=5)
            _ = plt.plot(xt4, yt4, 'o', label=f'XYt4 ({f_1f(xt4)}, {f_1f(yt4)})', markersize=5)





            ctx['xyo'] = [f_1f(xo), f_1f(yo)] or None
            ctx['xyt1'] = [f_1f(xt1), f_1f(yt1)] or None
            ctx['xyt2'] = [f_1f(xt2), f_1f(yt2)] or None
            ctx['xyt3'] = [f_1f(xt3), f_1f(yt3)] or None
            ctx['xyt4'] = [f_1f(xt4), f_1f(yt4)] or None
        else:
            ctx['error'] = True

        _ = plt.legend()

        # move x, y axes to center

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        ax.set_aspect('equal')
        ax.autoscale_view()

        ax.set_aspect(1.0)  # make aspect ratio square
        ax.grid(True)

        canvas = FigureCanvas(fig)
        outstr = io.BytesIO()
        canvas.print_png(outstr)
        inline_png = base64.b64encode(outstr.getvalue()).decode('ascii')
        outstr.close()

        # plt.show()
        ctx['distance'] = f_1f(distance)
        ctx['distance_centers'] = f_1f(distance_centers)
        ctx['inline_png'] = inline_png

        return ctx


    @staticmethod
    def fetch_values(form_data_list, keys):
        return [float(item['value']) for item in form_data_list if item['name'].startswith(keys)]

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        form_data = request.POST.get('formData')
        form_data_list = json.loads(form_data)

        # filter elements keep needed ones only
        form_data_list = [item for item in form_data_list if item['name'].startswith(('x_', 'y_', 'r_'))]

        c1_data, c2_data = self.fetch_values(form_data_list, ('x_1', 'y_1', 'r_1')), \
                           self.fetch_values(form_data_list, ('x_2', 'y_2', 'r_2'))

        ctx = dict()
        ctx['circle_1'] = c1_data
        ctx['circle_2'] = c2_data
        ctx.update(**self.get_results_context(c1_data, c2_data))

        return render(request, 'main/tangent_2_circles/includes/results_content.html', ctx)

    def get(self, request, *args, **kwargs):
        return render(request, 'main/tangent_2_circles/index.html')


class PolygonView(View):

    def to_convex_contour(self,
                          vertices_count,
                          x_generator=random.uniform,
                          y_generator=random.uniform):
        """
        Port of Valtr algorithm by Sander Verdonschot.

        Reference:
            http://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
        """
        xs = [x_generator(1.1, 10.3) for _ in range(vertices_count)]
        ys = [y_generator(1.2, 10.1) for _ in range(vertices_count)]
        xs = sorted(xs)
        ys = sorted(ys)
        min_x, *xs, max_x = xs
        min_y, *ys, max_y = ys
        vectors_xs = self._to_vectors_coordinates(xs, min_x, max_x)
        vectors_ys = self._to_vectors_coordinates(ys, min_y, max_y)
        random.shuffle(vectors_ys)

        def to_vector_angle(vector):
            x, y = vector
            return math.atan2(y, x)

        vectors = sorted(zip(vectors_xs, vectors_ys),
                         key=to_vector_angle)
        point_x = point_y = 0
        min_polygon_x = min_polygon_y = 0
        points = []
        for vector_x, vector_y in vectors:
            points.append((point_x, point_y))
            point_x += vector_x
            point_y += vector_y
            min_polygon_x = min(min_polygon_x, point_x)
            min_polygon_y = min(min_polygon_y, point_y)
        shift_x, shift_y = min_x - min_polygon_x, min_y - min_polygon_y
        return [(point_x + shift_x, point_y + shift_y)
                for point_x, point_y in points]

    def _to_vectors_coordinates(self, coordinates, min_coordinate, max_coordinate):
        last_min = last_max = min_coordinate
        result = []
        for coordinate in coordinates:
            if self._to_random_boolean():
                result.append(coordinate - last_min)
                last_min = coordinate
            else:
                result.append(last_max - coordinate)
                last_max = coordinate
        result.extend((max_coordinate - last_min,
                       last_max - max_coordinate))
        return result

    def _to_random_boolean(self):
        return random.getrandbits(1)

    @staticmethod
    def random_float(start=0.1, end=15.1):
        return random.uniform(start, end)

    @staticmethod
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        form_data = request.POST.get('formData')
        form_data_list = [item['value'] for item in json.loads(form_data) if item['name'] in {'points_count', 'lines_count'}]

        vertices_count = abs(int(form_data_list[0]))
        lines_count = abs(int(form_data_list[1]))

        fig, ax = plt.subplots()
        polygon_points = self.to_convex_contour(vertices_count)
        polygon = Polygon(polygon_points)
        x, y = polygon.exterior.xy
        ax.plot(x, y)

        # Multilines with shapely

        lines_xys = list(((self.random_float(), self.random_float()),
                          (self.random_float(), self.random_float())) for _ in range(lines_count) )
        # coords = [((0, 0), (1, 1)), ((-1, 0), (1, 0))]
        lines = MultiLineString(lines_xys)

        intersection_points = []
        mid_points = []

        for line in lines:
            # check if each line intersects with the polygon
            shapely_line = LineString(line)
            x, y = shapely_line.xy
            ax.plot(x, y)
            # Test intersection points with shapely
            intersection_point = list(polygon.intersection(shapely_line).coords)

            if len(intersection_point):
                mid_points.append(
                    self.midpoint(intersection_point[0], intersection_point[1])
                )

                intersection_points.extend(intersection_point)

        for polygon_point in polygon_points:
            label = f'({f_1f(polygon_point[0])}, {f_1f(polygon_point[1])})'
            plt.plot(polygon_point[0], polygon_point[1], 'o', label=label, markersize=3)

        if len(intersection_points):
            for point in intersection_points:
                label = f'({f_1f(point[0])}, {f_1f(point[1])})'
                plt.plot(point[0], point[1], 'o', label=label, markersize=5)

        if len(mid_points):
            for point in mid_points:
                label = f'({f_1f(point[0])}, {f_1f(point[1])})'
                plt.plot(point[0], point[1], 'o', label=label, markersize=5)

        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        ax.set_aspect('equal')
        ax.autoscale_view()

        ax.set_aspect(1.0)  # make aspect ratio square
        ax.grid(True)
        plt.legend()
        # plt.show()

        canvas = FigureCanvas(fig)
        outstr = io.BytesIO()
        canvas.print_png(outstr)
        inline_png = base64.b64encode(outstr.getvalue()).decode('ascii')
        outstr.close()

        ctx = {
            'vertices_count': vertices_count,
            'lines_count': lines_count,
            'polygon_points': [(f_1f(p[0]), f_1f(p[1])) for p in polygon_points],
            'intersection_points': [(f_1f(p[0]), f_1f(p[1])) for p in intersection_points],
            'inline_png': inline_png,

        }

        return render(request, 'main/polygon/includes/results_content.html', ctx)

    def get(self, request, *args, **kwargs):
        # Todo: use points as default, if possible
        # points = np.array([[1, 4.8], [2, 2], [4, 2], [5, 4.8], [3, 6]])

        return render(request, 'main/polygon/index.html')
