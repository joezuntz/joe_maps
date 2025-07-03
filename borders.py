import os
import geopandas as gpd

dirname = 1


class Borders:
    def __init__(self, path):
        self.path = path


def get_border_between(country1, country2, borders):
    b1 = borders.query(f"name=='{country1}'")
    b2 = borders.query(f"name=='{country2}'")
    g1 = b1.iloc[0].geometry
    g2 = b2.iloc[0].geometry
    return g1.boundary.intersection(g2.boundary)


def get_sea_border(country, neighbours, borders):

    border = borders.query(f"name=='{country}'").iloc[0].geometry.boundary

    for c in neighbours:
        other = borders.query(f"name=='{c}'").iloc[0].geometry.boundary
        border = border.difference(other)
    return border
