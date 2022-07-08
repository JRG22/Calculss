import numpy as np
import random as rd


# TODO mettre en commentaire en anglais


def calcul_N_manual(figure_noise, B):
    """
    Calcul of noise system
    """
    return 10 * np.log10(1.38e-23 * (figure_noise - 1) * 130 * B) + 30


class radioStation(object):

    def __init__(self, lat, long, alt, canal=0, p=40, f=5060e6):
        self.lat = lat  # latitude in degrees
        self.long = long  # longitude in degrees
        self.alt = alt  # altitude in meters
        self.coordinates = lat, long, alt  # Coordinates : np array
        self.pe = p  # Transmit power in dBm
        self.f = f  # Frequency use in hz
        self.ge = 0  # Antenna gain in dB
        self.fl = 0  # Feeder loss reception en dB
        self.noise = 0  # Noise in dBm
        self.B = 1  # Bandwidth in hz
        self.figure_noise = 1  # Figure Noise
        self.canal = canal  # channel use : int
        self.probability_message_lost = 0
        self.interf = None
        self.C = None
        self.criteria_c_i = None
        self.criteria_i_n = None
        self.limite_c_i = None
        self.limite_i_n = None
        self.N = -110

    def calcul_N(self):
        """
        Calcul of noise system
        """
        return 10 * np.log10(1.38e-23 * (self.figure_noise - 1) * 130 * self.B) + 30

    def set_random_coordinates(self, min, max):
        """ Randomize a caracteristic between min et max returning a float value"""
        self.lat = np.random.uniform(min, max)
        self.long = np.random.uniform(min, max)
        self.alt = np.random.uniform(min, max)


    def set_random_canal(self):
        """Modifie la valeur du canal utilisé de façon aléatoire """
        self.canal = np.floor(rd.uniform(1, 244))

    def set_coordinates(self):
        self.coordinates = self.lat, self.long, self.alt

    def set_canal(self, canal):
        """Modifie la valeur du numéro de canal utilisé

        Parameters
        ----------
        canal : int
            Numéro du canal d'émission
        Returns
        -------

        """
        self.canal = canal

    def get_parameters(self):
        """Retourne la puissance émise, le gain d'émission,les pertes de cables, le canal utilisé de la station de base

        Parameters
        ----------

        Returns
        -------
        float
            Puissance totale reçue en dB
        float
            Gain de l'antenne en dB
        float
            Feeder Loss
        int
            Numéro du canal d'émission

        """

        return self.pe, self.ge, self.fl, self.canal

    def get_coordinates(self, t=0):
        self.set_coordinates()
        return self.coordinates

    def set_criteria(self, criteria):
        self.criteria = criteria


    def __str__(self):
        return ("Coordonnées :  Longitude : " + str(self.long) + " Latitude :  " + str(self.lat) +
                "Num Canal : " + str(self.canal) + " Fréquence : " + str(self.f)+ "Bandwidth : " + str(self.B))
    def calcul_criterias(self):
        self.calcul_c_i()
        self.calcul_i_n()
    def calcul_c_i(self):

        self.criteria_c_i = self.C-self.interf

    def calcul_i_n(self):
        self.criteria_i_n  = self.interf-self.N


