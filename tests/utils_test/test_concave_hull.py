import bisect
from collections import OrderedDict
import math
#import numpy as np
import matplotlib.tri as tri
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import linemerge


class ConcaveHull:

    def __init__(self):
        self.triangles = {}
        self.crs = {}


    def loadpoints(self, points):
        #self.points = np.array(points)
        self.points = points


    def edge(self, key, triangle):
        '''Calculate the length of the triangle's outside edge
        and returns the [length, key]'''
        pos = triangle[1].index(-1)
        if pos==0:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][1]]
        elif pos==1:
            x1, y1 = self.points[triangle[0][1]]
            x2, y2 = self.points[triangle[0][2]]
        elif pos==2:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][2]]
        length = ((x1-x2)**2+(y1-y2)**2)**0.5
        rec = [length, key]
        return rec


    def triangulate(self):

        if len(self.points) < 2:
            raise Exception('CountError: You need at least 3 points to Triangulate')

        temp = list(zip(*self.points))
        x, y = list(temp[0]), list(temp[1])
        del(temp)

        triang = tri.Triangulation(x, y)

        self.triangles = {}

        for i, triangle in enumerate(triang.triangles):
            self.triangles[i] = [list(triangle), list(triang.neighbors[i])]


    def calculatehull(self, tol=50):

        self.tol = tol

        if len(self.triangles) == 0:
            self.triangulate()

        # All triangles with one boundary longer than the tolerance (self.tol)
        # is added to a sorted deletion list.
        # The list is kept sorted from according to the boundary edge's length
        # using bisect
        deletion = []
        self.boundary_vertices = set()
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, neigh in enumerate(triangle[1]):
                    if neigh == -1:
                        if pos == 0:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][1])
                        elif pos == 1:
                            self.boundary_vertices.add(triangle[0][1])
                            self.boundary_vertices.add(triangle[0][2])
                        elif pos == 2:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][2])
            if -1 in triangle[1] and triangle[1].count(-1) == 1:
                rec = self.edge(i, triangle)
                if rec[0] > self.tol and triangle[1].count(-1) == 1:
                    bisect.insort(deletion, rec)

        while len(deletion) != 0:
            # The triangles with the longest boundary edges will be
            # deleted first
            item = deletion.pop()
            ref = item[1]
            flag = 0

            # Triangle will not be deleted if it already has two boundary edges
            if self.triangles[ref][1].count(-1) > 1:
                continue

            # Triangle will not be deleted if the inside node which is not
            # on this triangle's boundary is already on the boundary of
            # another triangle
            adjust = {0: 2, 1: 0, 2: 1}
            for i, neigh in enumerate(self.triangles[ref][1]):
                j = adjust[i]
                if neigh == -1 and self.triangles[ref][0][j] in self.boundary_vertices:
                    flag = 1
                    break
            if flag == 1:
                continue

            for i, neigh in enumerate(self.triangles[ref][1]):
                if neigh == -1:
                    continue
                pos = self.triangles[neigh][1].index(ref)
                self.triangles[neigh][1][pos] = -1
                rec = self.edge(neigh, self.triangles[neigh])
                if rec[0] > self.tol and self.triangles[rec[1]][1].count(-1) == 1:
                    bisect.insort(deletion, rec)

            for pt in self.triangles[ref][0]:
                self.boundary_vertices.add(pt)

            del self.triangles[ref]

        self.polygon()



    def polygon(self):

        edgelines = []
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, value in enumerate(triangle[1]):
                    if value == -1:
                        if pos==0:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][1]]
                        elif pos==1:
                            x1, y1 = self.points[triangle[0][1]]
                            x2, y2 = self.points[triangle[0][2]]
                        elif pos==2:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][2]]
                        line = LineString([(x1, y1), (x2, y2)])
                        edgelines.append(line)

        bound = linemerge(edgelines)

        self.boundary = Polygon(bound.coords)

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

gdf = gpd.read_file(os.path.join("..", "..", "dto", "raspadskaya", "input", "vector", "surface_quarry2023.shp"))
# gdf_s = gdf.dissolve()
# ch = gdf_s.geometry.concave_hull(allow_holes=False)
# ch.to_file(os.path.join("..", "..", "dto", "raspadskaya", "input", "vector", "drain2023.shp"))
pts = gdf.get_coordinates().to_numpy()
ch = ConcaveHull()
ch.loadpoints(pts)
ch.calculatehull()
boundary_points = np.vstack(ch.boundary.exterior.coords.xy).T
conv_gdf = gpd.GeoDataFrame(geometry=[Polygon(boundary_points)])
conv_gdf.to_file(os.path.join("..", "..", "dto", "raspadskaya", "input", "vector", "drain2023.shp"))
