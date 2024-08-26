import numpy as np

def random_walk(n_steps):
    steps = np.random.choice([-1, 1], size=n_steps)
    return np.cumsum(steps)


def get_random_walk_frequency(n_steps, frequency_base, walk_scale):
    walk = random_walk(n_steps)
    return frequency_base - walk_scale * (walk - walk.mean())


def get_chirp_signal(duration, 
                     f_start,
                     f_end,
                     sampling_frequency):
    
    time_vector = np.linspace(0, duration, int(duration * sampling_frequency))

    # Compute the phase by integrating the frequency
    phase_t = 2 * np.pi * (f_start + (f_end - f_start) * time_vector / (2 * duration)) * time_vector

    # Generate the chirp signal
    return np.sin(phase_t)


def get_random_walk_chirp_signal(duration, 
                                 random_walk_freq, 
                                 sampling_frequency):
    
    n_steps = len(random_walk_freq)
    
    sub_duration =  duration / (n_steps - 1)
    sub_n_steps = int(sampling_frequency * sub_duration)
    
    prev_phase_t = 0
    signal = []
    for i in range(n_steps - 1):
        f_start, f_end = random_walk_freq[i: i + 2]
        
        slope = (f_end - f_start) / sub_duration
        
        times = np.linspace(sub_duration * i, 
                            sub_duration * (i + 1), 
                            sub_n_steps + 1)
        
        phase_t = prev_phase_t + f_start * (times - i * sub_duration) + (slope / 2) * (times - i * sub_duration) ** 2
        prev_phase_t = phase_t[-1]
        
        sub_signal = np.sin(2 * np.pi * phase_t[:-1])

        signal.append(sub_signal)
    
    return np.hstack(signal)
