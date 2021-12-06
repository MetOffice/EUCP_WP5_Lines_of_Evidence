from ascend import shape

# Script to create EUCP WP3 shapefiles.
# Coordinates taken from
# https://wiki.eucp-project.eu/xwiki/bin/view/WP3/Simulation%20protocol/
# ALP-3 domain taken from Pichelli (2021)
# Tested / runs with /net/project/ukmo/scitools/opt_scitools/environments/default/2021_03_18-1/bin/python

corners = {}

corners['NWE-3'] = [(-8, 40.4), (11, 40.4), (15.2, 58.6), (-12.5, 58.6)]
corners['SWE-3'] = [(-10, 30), (7.4, 33), (5.7, 48.9), (-15, 45.5)]
corners['SEE-3'] = [(12.5, 34.3), (28.5, 34.3), (29.4, 40.9), (11.5, 40.9)]
corners['CEU-3'] = [(4.6, 44.6), (18.5, 45.5), (18.7, 56.5), (1.2, 55.4)]
corners['CEE-3'] = [(17.8, 41.5), (31.3, 41.5), (32.8, 51.6), (16.4, 51.6)]
corners['NEU-3'] = [(1, 50.7), (26.7, 49.7), (44.1, 70.6), (-9.4, 72.6)]
corners['ALP-3'] = [(1, 40), (17, 40), (17, 50), (1, 50)]

shps = shape.ShapeList()

for k, v in corners.items():
    shp = shape.create(v, {'shape': 'rectangle'}, 'Polygon')
    shp.attributes['NAME'] = k
    shps.append(shp)

shape.save_shp(shps, 'EUCP_WP3_domains', 'polygon')
