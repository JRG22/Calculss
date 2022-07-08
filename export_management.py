import csv
import os.path

import numpy as np
import fonctions_generales as fg


def export_caracteristics(drone, base_station):
    """Exporte les caractéristiques des

        Parameters
        ----------
        pe_dbm: float
            Puissance transmise en dBm
        nb_canal_e : int[]
            Canal ou canaux de ou des émetteurs
        nb_canal_r : int []
            Canal ou canaux de ou des récepteurs
        affichage : boolean
            show  graphics of masks and filtred power
        Returns
        -------
        float[]
            Tableau des Puissances filtrées en dBm
        """
    datas = drone.get_parameters()
    datas = np.append(datas, base_station.get_parameters())
    with open('./Results/char_file.csv', mode='w', newline='') as char_file:
        char_file = csv.writer(char_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        char_file.writerow(
            ['Drone : Puissance émise', 'Gain de l\'antenne', 'feeder Loss', 'Ground station : Puissance émise ',
             'Gain ', 'Gain de l\'antenne', 'feeder Loss', ])
        char_file.writerow(datas)
        # En dessous en commentaire si on veut mettre les coordonnées mais pas sur que ce soit une bonne idée.
        """for i in range(int(np.size(drone.trajectoire) / 3)):
            datas = np.append(datas, fg.calcul_distance_grand_cercle(np.radians(base_station.get_coordinates()),
                                                                     np.radians(drone.trajectoire[:, i])))"""


def export_char_trajectoire(drone):
    traj = drone.trajectoire_obj
    datas = traj.coordonnees

    with open('./Results/char_traj.csv', mode='w', newline='', ) as char_trajectoire:
        char_trajectoire = csv.writer(char_trajectoire, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        char_trajectoire.writerow(
            ['Latitude (degrés)', 'Longitude (degrés)', 'Altitude (mètres)', 'Service range(mètres)',
             'Distance de séparation (mètres)', 'Angle ground station (degrés',
             'puissance reçue (dbm)', 'distance_interferer', 'Interférence (dbm)'])
        i = 0
        while i < np.size(datas[0]):
            lat = float(traj.coordonnees[:, i][0])
            long = float(traj.coordonnees[:, i][1])
            alt = float(traj.coordonnees[:, i][2])
            char_trajectoire.writerow([lat, long, alt, float(drone.trajectoire_obj.service_range[i]),
                                       drone.trajectoire_obj.distance_ground_station[i],
                                       float(drone.trajectoire_obj.angle_ground_station[i]),
                                       drone.trajectoire_obj.pr_drone[i], drone.trajectoire_obj.distance_Interferer[i],
                                       drone.trajectoire_obj.NplusI[i]])
            i += 1


def export_coord_trajectoire(drone):
    datas = drone.trajectoire
    with open('./Results/trajectoire.csv', mode='a', newline='', ) as trajectoire:
        trajectoire = csv.writer(trajectoire, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if os.path.getsize('trajectoire.csv') == 0: trajectoire.writerow(['Latitude', 'Longitude', ' Altitude'])
        i = 0
        while i < (np.size(datas) / 3):
            trajectoire.writerow(datas[:, i])
            i += 1


def export_puissance_recues(drone):
    with open('./Results/puissance_recues.csv', mode='w', newline='', ) as puissance_recues:
        trajectoire = csv.writer(puissance_recues, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectoire.writerow(drone.trajectoire_obj.pr_drone)


def clear_file():
    with open('./Results/trajectoire.csv', mode='w', newline='') as trajectoire:
        trajectoire = csv.writer(trajectoire, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectoire.writerow(['Latitude', 'Longitude', ' Altitude'])


def export_service_range(drone, base_station):
    with open('./Results/service_range.csv', mode='w', newline='') as service_range:
        service_range = csv.writer(service_range, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        coord_drone_sol_rad = np.array([np.radians(drone.trajectoire[0, :]), np.radians(drone.trajectoire[1, :]), 30],
                                       dtype=object)
        d = fg.calcul_distance_grand_cercle(coord_drone_sol_rad, np.radians(base_station.get_coordinates()))

        service_range.writerow(d)


def export_summary_(drone, ground_station, satellite):
    with open('./Results/summary.csv', mode='w', newline='') as summary:
        summary = csv.writer(summary, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        summary.writerow(
            ['Temps en minutes', 'Critères C/I : UA -> UA terrestrial', 'UA -> UA Satellite ', 'UA -> ground station',
             'UA -> Satellite',
             ' Critères I/N  : UA -> UA ', 'UA -> ground station', 'UA -> Satellite'])
        for i in range(np.size(drone.trajectoire_obj.temps)):
            summary.writerow(
                [drone.trajectoire_obj.temps[i], drone.criteria_c_i[i], drone.criteria_c_i[i],
                 ground_station.criteria_c_i[i], satellite.criteria_c_i[i], drone.criteria_c_i[i],
                 ground_station.criteria_i_n[i], satellite.criteria_i_n[i]])
