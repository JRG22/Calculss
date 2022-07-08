import pandas as pd

from ANFRpy.geometry.ENU_to_ECEF import *
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import geog
from cartopy.io.img_tiles import OSM
import shapely.geometry
from cartopy.geodesic import Geodesic


def carte_test(drone, base_station, tab_drone_i):
    coordonnees_drone = drone.trajectoire_obj.coordonnees
    latitude = coordonnees_drone[0].astype(float)
    longitude = coordonnees_drone[1].astype(float)
    plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax = plt.axes(projection=ccrs.EquidistantConic(2, 48))

    ax.set_extent([0, 4, 47, 49])
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)

    ax.gridlines(draw_labels=True, xlocs=np.arange(0, 4, 0.5), ylocs=np.arange(47, 49, 0.5))
    # imagery = OSM()
    # ax.add_image(imagery, )
    # plus c'est grand, plus c'est précis, plus ça prend du temps

    ax.set_title('Geographic scenario of current studies')
    ax.scatter(longitude, latitude, alpha=1, c='b', s=10, label="Victim drone trajectory", transform=ccrs.PlateCarree())
    for drone_i in tab_drone_i:
        coordonnees_drone_i = drone_i.trajectoire_obj.coordonnees
        latitude_i = coordonnees_drone_i[0].astype(float)
        longitude_i = coordonnees_drone_i[1].astype(float)
        ax.scatter(longitude_i, latitude_i, zorder=1, alpha=1, c='g', s=10,
                   label="Interfering drone trajectory", transform=ccrs.PlateCarree())
    ax.scatter(base_station.get_coordinates()[1], base_station.get_coordinates()[0], alpha=1, c='r', s=10,
               label="Ground station", transform=ccrs.PlateCarree())
    gd = Geodesic()
    cercle = gd.circle(lon=2, lat=48, radius=100000)
    ax.plot(cercle[:, 0], cercle[:, 1], alpha=1, c='black',
            label="Service range of 100 km", transform=ccrs.PlateCarree())
    plt.legend(loc="upper left")



def carte(drone, base_station, tab_drone_i):
    coordonnees_drone = drone.trajectoire_obj.coordonnees
    latitude = coordonnees_drone[0].astype(float)
    longitude = coordonnees_drone[1].astype(float)

    """BBox = ((longitude.min(), longitude.max(),
             latitude.min(), latitude.max()))"""

    carte = plt.imread("C:\\Users\ROUSSEL-GALLE\Pictures\carte.png")
    fig, ax = plt.subplots(figsize=(8, 8))

    for drone_i in tab_drone_i:
        coordonnees_drone_i = drone_i.trajectoire_obj.coordonnees
        latitude_i = coordonnees_drone_i[0].astype(float)
        longitude_i = coordonnees_drone_i[1].astype(float)
        ax.scatter(longitude_i, latitude_i, zorder=1, alpha=1, c='g', s=15,
                   label="trajectoire du drone interférent")

    ax.scatter(longitude, latitude, zorder=1, alpha=1, c='b', s=10, label=" trajectoire du drone")
    ax.scatter(base_station.get_coordinates()[1], base_station.get_coordinates()[0], zorder=1, alpha=1, c='r', s=20,
               label="Station de base")

    ax.set_title('Trajectoire rectiligne de drones en LOS de la Base station')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_ylim(47, 49)
    BBox = (0, 4, 47, 49)
    draw_circle = plt.Circle((2, 48), 1.1, fill=False, label="Zone de couverture = 100km")
    ax.set_aspect(1)
    ax.add_artist(draw_circle)
    plt.legend(loc='upper left')
    ax.imshow(carte, zorder=0, extent=BBox, aspect='equal')


def carte2(base_station):
    df = pd.read_csv('E:\ANFR\8-Apprentissage\\5 - Période 5\Conception Projet\Calculss\\trajectoire.csv', sep=';',
                     header=0, encoding='ascii', engine='python')
    df['Latitude'] = df['Latitude'].apply(lambda x: x.split()[0].replace('[', ''))
    df['Latitude'] = df['Latitude'].apply(lambda x: x.split()[0].replace(']', ''))
    df['Longitude'] = df['Longitude'].apply(lambda x: x.split()[0].replace('[', ''))
    df['Longitude'] = df['Longitude'].apply(lambda x: x.split()[0].replace(']', ''))
    df = df.astype({'Latitude': 'float', 'Longitude': 'float'})
    latitude = df['Latitude']
    longitude = df['Longitude']

    BBox = ((longitude.min(), longitude.max(),
             latitude.min(), latitude.max()))



    carte = plt.imread("C:\\Users\ROUSSEL-GALLE\Pictures\carte.png")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(longitude, latitude, zorder=1, alpha=1, c='b', s=10, label=" trajectoire du drone")

    ax.scatter(base_station.get_coordinates()[1], base_station.get_coordinates()[0], zorder=1, alpha=1, c='r', s=20,
               label="Station de base")

    ax.set_title('Trajectoire rectiligne de drones en LOS de la Base station')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_ylim(47, 49)
    BBox = ((0, 4, 47, 49))
    draw_circle = plt.Circle((2, 48), 1.1, fill=False, label="Zone de couverture = 100km")

    ax.set_aspect(1)
    ax.add_artist(draw_circle)
    plt.legend(loc='upper left')

    ax.imshow(carte, zorder=0, extent=BBox, aspect='equal')
