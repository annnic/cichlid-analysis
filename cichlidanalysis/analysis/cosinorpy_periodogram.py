import numpy as np

import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib

import os

# adapted from https://github.com/mmoskon/CosinorPy/blob/master/CosinorPy/cosinor.py to make the figure


def periodogram_df_an(df, folder='', prefix='', **kwargs):
    names = list(df.test.unique())
    names.sort()

    for name in names:
        x, y = df[df.test == name].x.values, df[df.test == name].y.values
        if folder:
            save_to = os.path.join(folder, "per_" + name)
        else:
            save_to = ""

        periodogram_an(x, y, save_to=save_to, name=name, prefix=prefix, **kwargs)


def periodogram_an(X, Y, per_type='per', sampling_f='', logscale=False, name='', save_to='', prominent=False,
                   max_per=240, prefix=''):
    SMALLEST_SIZE = 5
    SMALL_SIZE = 6
    matplotlib.rcParams.update({'font.size': SMALLEST_SIZE})

    if per_type == 'per' or per_type == 'welch':

        X_u = np.unique(X)
        Y_u = []
        for x_u in X_u:
            # y_u.append(np.mean(y[t == x]))
            Y_u.append(np.median(Y[x_u == X]))

        if not sampling_f:
            sampling_f = 1 / (X[1] - X[0])

        Y = Y_u

    if per_type == 'per':
        # Fourier
        f, Pxx_den = signal.periodogram(Y, sampling_f)
    elif per_type == 'welch':
        # Welch
        f, Pxx_den = signal.welch(Y, sampling_f)
    elif per_type == 'lombscargle':
        # Lomb-Scargle
        min_per = 2
        # max_per = 50

        f = np.linspace(1 / max_per, 1 / min_per, 10)
        Pxx_den = signal.lombscargle(X, Y, f)
    else:
        print("Invalid option")
        return

    # significance
    # Refinetti et al. 2007
    p_t = 0.05

    N = len(Y)
    T = (1 - (p_t / N) ** (1 / (N - 1))) * sum(Pxx_den)  # threshold for significance

    if f[0] == 0:
        per = 1 / f[1:]
        Pxx = Pxx_den[1:]
    else:
        per = 1 / f
        Pxx = Pxx_den

    Pxx = Pxx[per <= max_per]
    per = per[per <= max_per]

    plt.figure(figsize=(1.5, 1.5))
    try:
        if logscale:
            plt.semilogx(per, Pxx, 'ko', markersize=1)
            plt.semilogx(per, Pxx, 'k-', linewidth=0.5)
            plt.semilogx([min(per), max(per)], [T, T], 'k--', linewidth=0.5)
        else:
            plt.plot(per, Pxx, 'ko', markersize=1)
            plt.plot(per, Pxx, 'k-', linewidth=0.5)
            plt.plot([min(per), max(per)], [T, T], 'k--', linewidth=0.5)
    except:
        print("Could not plot!")
        return

    peak_label = ''

    if prominent:
        locs, heights = signal.find_peaks(Pxx, height=T)

        if any(locs):
            heights = heights['peak_heights']
            s = list(zip(heights, locs))
            s.sort(reverse=True)
            heights, locs = zip(*s)

            heights = np.array(heights)
            locs = np.array(locs)

            peak_label = ', max peak=' + str(per[locs[0]])

    else:
        locs = Pxx >= T
        if any(locs):
            heights, locs = Pxx[locs], per[locs]
            HL = list(zip(heights, locs))
            HL.sort(reverse=True)
            heights, locs = zip(*HL)

            peak_label = ', peaks=\n'

            locs = locs[:11]
            for loc in locs[:-1]:
                peak_label += "{:.2f}".format(loc) + ','
            peak_label += "{:.2f}".format(locs[-1])

    plt.xlabel('period [hours]')
    plt.ylabel('PSD')
    plt.title(name + peak_label, fontsize=SMALLEST_SIZE)

    # Adjust the offset for tick labels on all axes
    plt.tick_params(axis='x', pad=0.5, length=2)
    plt.tick_params(axis='y', pad=0.5, length=2)

    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0, 12, 24, 48])

    # plt.tight_layout()

    if save_to:
        plt.savefig(save_to + prefix + '.pdf')
        plt.tight_layout()
        plt.savefig(save_to + prefix + '_tight.pdf')
        # plt.savefig(save_to + prefix + '.png')
        plt.close()
    else:
        plt.show()
    return
