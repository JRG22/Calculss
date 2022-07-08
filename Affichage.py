"""Classe affichage possède plusieurs fonctions d'affichage 2d,3d, de C, C/N+I en spatial..."""

import matplotlib.pyplot as plt
import numpy as np
from radioStation import *
import UAS

plt.rcParams['axes.grid'] = True

# figgg, axis = plt.subplots(2, 2)

N = calcul_N_manual(7, 1)


def display_text(c, i):
    """Affiche C, I et C/I

        Parameters
        ----------
        c : float
            Puissance de réception voulue en dBm
        i : float
            Puissance de réception en dBm

        Returns
        -------
        
        """

    print("Puissance reçue voulue : ", c)

    print("Puissance reçue non voulue : ", i)
    res = c - 10 * np.log10(10 ** (N / 10) + 10 ** (i / 10))

    print("C/N+I :", res)


def affichage_scenario1(nb_channel_sr, distances):
    """Retourne la puissance totale reçue par le drone victime

            Parameters
            ----------
            nb_channel_sr : float[]

            distances : float[]
                Distances requises en mètres pour chaque canal
            Returns
            -------
            float
                Puissance totale reçue en dB
            float[]
                Tableau de chaques puissances indépendamment reçues
            """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nb_channel_sr, distances, 'ro')
    ax.set_yscale('log')
    plt.xticks(np.arange(min(nb_channel_sr), max(nb_channel_sr), step=1))
    plt.xlabel('Décalage fréquentiel en nombre de canaux')
    plt.ylabel('Distance requise en mètres')
    plt.title("Distance minimale pour un I/N de -6dB")
    for i in range(len(nb_channel_sr)):
        plt.annotate(round(distances[i], 1), (nb_channel_sr[i], distances[i]))
    plt.grid(True, which="both")
    plt.show()



def affichage_scenario3(distance, pr1):
    """affiche le résultat du scénario3

        Parameters
        ----------

        Returns
         -------


        """
    plt.plot(distance, pr1, color="black", label="valeurs calculées de puissances totale reçues")
    plt.plot(distance, np.ones(len(pr1)) * -119, color='red', label='sensibilité du récepteur -119 dbm')

    plt.legend()

    plt.xlabel('distance séparant les drones en mètres')
    plt.ylabel('puissance totale reçue par le drone en dbm')

    plt.grid()
    plt.show()


def affichage_scenario6(mat_results, channels):
    """

    :param channels: UAS altitude array
    :param mat_results: Resulting matrice
    :return:
    """
    # TODO : Legende titre graphique etc
    fig, axes = plt.subplots()
    axes.plot(channels, mat_results, marker="x")
    plt.show()


def plot_cdf_assembles(a, titre, ligne, colonne):
    # TODO:voir pour les axis + description de la fonction
    pr_cumulative = np.array([])
    prob = np.array([])
    mini = np.amin(a)
    for k in range(100):
        nb = 0
        for i in range(len(a)):
            if a[i] < mini:
                nb += 1
        pr_cumulative = np.append(pr_cumulative, mini)
        prob = np.append(prob, 1 - nb / len(a))
        mini += np.abs(np.amax(a) - np.amin(a)) / 100

    axis[ligne, colonne].plot(pr_cumulative, prob, color='black')
    axis[ligne, colonne].set_yscale('log')
    axis[ligne, colonne].title.set_text(titre)

    plt.grid(True)

def plot_cdf(valeurs):
    fig,ax = plt.subplots()
    pr_cumulative = np.array([])
    prob = np.array([])
    mini = np.amin(valeurs)
    for k in range(100):
        nb = 0
        for i in range(len(valeurs)):
            if valeurs[i] < mini:
                nb += 1
        pr_cumulative = np.append(pr_cumulative, mini)
        prob = np.append(prob, 1 - nb / len(valeurs))
        mini += np.abs(np.amax(valeurs) - np.amin(valeurs)) / 100
    ax.plot(pr_cumulative, prob, color='black')
    ax.set_yscale('log')
    print("mini", np.amin(valeurs), "max", np.amax(valeurs))
    ax.set_xlim([np.amin(valeurs), np.amax(valeurs)])
    ax.grid(True)


class Affichage:
    """class affichage 3d plot scatter"""

    def __init__(self):
        """Instancie les attributs d'affichage, latitudes, longitudes, carrier power, interference, Critère de
        protection """
        self.lat = np.array([])
        self.long = np.array([])
        self.alt = np.array([])
        self.c = np.array([])
        self.i = np.array([])
        self.criteria = np.array([])

    def add_long(self, x):
        """Ajoute une valeur de longitude au tableau corrrespondant"""
        self.long = np.append(self.long, x)

    def add_lat(self, x):
        """Ajoute une valeur de latitude au tableau corrrespondant"""
        self.lat = np.append(self.lat, x)

    def add_alt(self, x):
        """Ajoute une valeur de latitude au tableau corrrespondant"""
        self.alt = np.append(self.alt, x)

    def add_pr(self, x):
        """Ajoute une valeur de C au tableau corrrespondant"""
        self.c = np.append(self.c, x)

    def add_i(self, x):
        """Ajoute une valeur d'interférence I au tableau corrrespondant"""

        self.i = np.append(self.i, x)

    def plot_3d(self, coord_drone, coord_bs):
        """Affiche le scénario en 3d"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.lat, self.long, self.alt, marker="o")
        ax.scatter(coord_drone[0], coord_drone[1], coord_drone[2], label="drone Victime", marker="x")
        ax.scatter(coord_bs[0], coord_bs[1], coord_bs[2], label="Station de base", marker="x")
        plt.title("Représentation en 3D")
        plt.legend(loc="lower right")
        ax.set_xlabel('Longitude en degrés')
        ax.set_ylabel('Latitude en degrés')
        ax.set_zlabel('Altitude en mètres')

    def plot_c_3d(self):
        """Affiche C en fonction de la longitude/latitude en 3d"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.lat, self.long, self.c)
        plt.title("Carrier power received")
        ax.set_xlabel('Longitude en degrés')
        ax.set_ylabel('Latitude en degrés')
        ax.set_zlabel('C en dBm reçue par le drone aux coordonnées correspondantes')

        plt.show()

    def plot_criteria_3d(self, titre):
        """Affiche le critère de protection en fonction de la longitude/latitude en 3d"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sp = ax.scatter(self.lat, self.long, self.alt, s=10, c=self.criteria, cmap="jet")
        plt.colorbar(sp, label="C/N+I en dB")
        plt.title(titre)

        ax.set_xlabel('Longitude en degrés')
        ax.set_ylabel('Latitude en degrés')

        plt.show()

    def plot_c_2d(self):
        """Affiche C en fonction de la longitude/latitude en 2d"""

        fig, ax = plt.subplots()
        sp = ax.scatter(self.lat, self.long, s=20, c=self.c, cmap="jet")
        plt.title("Carrier power received")

        fig.colorbar(sp, label="C en dBm")

        plt.show()

    def plot_criteria_2d(self, coord_element):
        """Affiche le critère de protection en fonction de la longitude/latitude en 2d"""

        fig, ax = plt.subplots()
        sp = ax.scatter(self.long, self.lat, s=10, c=self.criteria, cmap="jet")
        ax.scatter(coord_element[1], coord_element[0], label="Drone victime")
        plt.title("Critère de protection au niveau du drone")
        fig.colorbar(sp, label="C/N+I en dB")
        ax.set_xlabel('Longitude en degrés')
        ax.set_ylabel('Latitude en degrés')

    def set_criteria_csuri(self, c, i):
        """Calcule le critère de protection C/N+I à partir de C,N et I"""

        self.criteria = np.squeeze(c - 10 * np.log10(10 ** (N / 10) + 10 ** (i / 10)))

    def set_criteria_isurn(self, inter):
        """Calcule le critère de protection IsurN à partir de I et N"""

        self.criteria = inter - N

    def plot_cdf_csuri(self):

        """Affiche la fonction de répartition du critère de protection c sur i"""
        print("Criteria", self.criteria)
        pr_cumulative = np.array([])
        prob = np.array([])
        mini = np.amin(self.criteria)
        pas = np.abs((np.amax(self.criteria) - (np.amin(self.criteria) - 0.5))) / 100
        for k in range(100):
            nb = 0
            for i in range(len(self.criteria)):
                if self.criteria[i] <= mini:
                    nb += 1

            pr_cumulative = np.append(pr_cumulative, mini)
            prob = np.append(prob, 1 - nb / len(self.criteria))
            mini += pas

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(pr_cumulative, prob, color='black')
        ax.set_yscale('log')
        # plt.plot(np.array([0, 0]), np.array([0, 1]), color="red") --> Limite criteria
        plt.title("Fonction de répartition du critère C/N+I")
        plt.xlabel("Critère de protection en dB")
        plt.ylabel("Pb d'obtention d'une valeur inférieure au critère")
        # plt.xlim(-25, 5) --> Limites du graphique

        plt.grid(True)

    def plot_cdf_isurn(self):
        """Affiche la fonction de répartition du critère de protection isurn"""

        pr_cumulative = np.array([])
        prob = np.array([])
        mini = np.amin(self.criteria)
        for k in range(100):
            nb = 0
            for i in range(len(self.criteria)):

                if self.criteria[i] < mini:
                    nb += 1
            pr_cumulative = np.append(pr_cumulative, mini)
            prob = np.append(prob, 1 - nb / len(self.criteria))
            mini += (np.amax(self.criteria) - (np.amin(self.criteria) - 0.5)) / 100

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(pr_cumulative, prob, color='black')
        ax.set_yscale('log')
        plt.plot(np.array([-6, -6]), np.array([0, 1]), color="red")
        plt.title("Fonction de répartition du critère I/N")
        plt.xlabel("Critère de protection en dB")
        plt.ylabel("Pb d'obtention d'une valeur inférieure au critère")
        # plt.xlim(-18, 10) --> Limite du graphique

        plt.grid(True)


    def plot_all(self, affiche, coord_point):
        """Appelle les fonctions d'affichage"""
    #TODO : Supprimer cette fonction ?
        if affiche:
            self.plot_cdf_isurn()
            # self.plot_cdf_csuri()
            self.plot_criteria_2d(coord_point)
            self.plot_3d(coord_point)
            self.plot_criteria_3d()
            figgg.tight_layout()
            plt.show()
