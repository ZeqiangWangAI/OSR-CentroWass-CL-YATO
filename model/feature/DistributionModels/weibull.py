"""
Author: Terrance E. Boult.
This class extends the default libmr and aims to provide more useful curve fitting options.
"""
import torch
from .libmrTorch import libmr as libmrTorch


class weibull(libmrTorch):
    # TB's new mode that flips the data around the max and then does a fit low reject so that its effectively modeling just above the max .
    def FitLowNormalized(self, data, tailSize, isSorted=False, gpu=0):
        maxval = data.max(dim=1).values
        # Flip the data around the max so the smallest points are just beyond the data but mirrors the distribution
        data = 2 * maxval - data
        # Because of the flipping of the data, probability of unknown is returned
        return self.FitLow(data, tailSize, isSorted=isSorted, gpu=gpu)

    def pdf(self, distances):
        """
        This function can calculate raw probability scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        print("The pdf method is experimental and its functionality is not officially confirmed yet")
        weibulls, distances = self.compute_weibull_object(distances)
        return torch.exp(weibulls.log_prob(distances))
