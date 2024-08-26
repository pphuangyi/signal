#!/usr/bin/env python
import numpy as np
from torch.utils.data import Dataset


class DatasetConst(Dataset):
    def __init__(self,
                 duration,
                 sampling_frequency,
                 num_samples,
                 frequency_lo,
                 frequency_hi,
                 noise_std_lo,
                 noise_std_hi,
                 signal_mag=1):

        self.num_samples  = num_samples
        self.frequency_lo = frequency_lo
        self.frequency_hi = frequency_hi
        self.noise_std_lo = noise_std_lo
        self.noise_std_hi = noise_std_hi
        self.signal_mag   = signal_mag

        # time vector
        time_steps = int(duration * sampling_frequency)
        self.times = np.linspace(0, duration, time_steps + 1)[:-1]

    def __len__(self, ):
        return self.num_samples

    def __getitem__(self, index):
        frequency = np.random.uniform(self.frequency_lo, self.frequency_hi)
        noise_std = np.random.uniform(self.noise_std_lo, self.noise_std_hi)

        phase_t = self.get_phase(frequency)

        # generate signal
        signal = self.signal_mag * np.sin(phase_t)

        # add noise
        signal += noise_std * np.random.randn(len(signal))

        return signal, frequency, noise_std

    def get_phase(self, frequency):
        # get phase as a function of time
        phase_t = 2 * np.pi * frequency * self.times

        return phase_t


class DatasetRandomWalk(Dataset):
    def __init__(self,
                 duration,
                 sampling_frequency,
                 num_samples,
                 random_walk_steps,
                 random_walk_mag,
                 frequency_base,
                 noise_std_lo,
                 noise_std_hi,
                 signal_mag=1):

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.num_samples = num_samples

        self.random_walk_steps = random_walk_steps
        self.random_walk_mag = random_walk_mag
        self.frequency_base = frequency_base
        self.noise_std_lo = noise_std_lo
        self.noise_std_hi = noise_std_hi
        self.signal_mag = signal_mag

    def __len__(self, ):
        return self.num_samples

    def __getitem__(self, index):

        noise_std = np.random.uniform(self.noise_std_lo, self.noise_std_hi)

        random_walk_frequencies = self.get_random_walk_frequency()
        phase_t = self.get_phase(random_walk_frequencies)

        # generate signal
        signal = self.signal_mag * np.sin(phase_t)

        # add noise
        signal += noise_std * np.random.randn(len(signal))

        return signal, random_walk_frequencies, noise_std

    def get_random_walk_frequency(self, ):

        size = self.random_walk_steps + 1
        random_walk = np.cumsum(np.random.choice([-1, 1], size=size))
        shift_random_walk = random_walk - random_walk.mean()

        random_walk_frequencies = (self.random_walk_mag * shift_random_walk
                                   + self.frequency_base)
        assert random_walk_frequencies.min() > 0, \
            "minimum frequency must be positive"

        return random_walk_frequencies

    def get_phase(self, random_walk_frequencies):

        sub_duration =  self.duration / self.random_walk_steps
        sub_n_steps = int(self.sampling_frequency * sub_duration)

        times = np.linspace(0, sub_duration, sub_n_steps + 1)

        prev_phase = 0
        phase_t = []
        for i in range(self.random_walk_steps):
            f_start, f_end = random_walk_frequencies[i: i + 2]

            slope = (f_end - f_start) / sub_duration

            sub_phase_t = (prev_phase + f_start * times
                           + (slope / 2) * times ** 2)

            prev_phase = sub_phase_t[-1]
            phase_t.append(sub_phase_t[:-1])

        return 2 * np.pi * np.hstack(phase_t)
