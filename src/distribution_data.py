from dataclasses import dataclass

import numpy as np


@dataclass
class DistributionData:
    def __init__(self, q1, q3, med, lower_bound, upper_bound):
        self.q1 = q1
        self.q3 = q3
        self.med = med
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        class_name = self.__class__.__name__
        values = ', '.join([f'{item[0]}={round(item[1], 6)}' for item in self.__dict__.items()])

        return f'{class_name}({values})'


def get_distribution_data(distances):
    q1 = np.quantile(distances, 0.25)
    q3 = np.quantile(distances, 0.75)

    med = np.median(distances)

    # interquartile range
    iqr = q3 - q1

    # finding upper and lower whiskers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return DistributionData(q1, q3, med, lower_bound, upper_bound)
