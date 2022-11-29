from utils_func import *

"""# Wavelet"""


def showWaveletFrequency(channel, isShow=False):
    if isShow:
        titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate(channel):  # [LL, LH, HL, HH]
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.show()


def getWaveletFrequency(frame, wavelet_family):
    # https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
    original = copy.deepcopy(frame)
    # original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    R = original[:, :, 0]
    # sym3 periodization
    # haar periodization
    # db1 symmetric
    mode = "symmetric"
    if "sym" in wavelet_family:
        mode = "periodization"
    wp = pywt.WaveletPacket2D(R, wavelet=wavelet_family,
                              mode=mode)
    # LL, (LH, HL, HH) = wp['a'].data, wp['h'].data, wp['v'].data, wp['d'].data
    a = None
    d = None
    v = None
    h = None

    try:
        a = wp['a']
    except:
        a = None
    try:
        h = wp['h']
    except:
        h = None
    try:
        v = wp['v']
    except:
        v = None
    try:
        d = wp['d']
    except:
        d = None

    LL = [] if a is None else a.data
    LH = [] if h is None else h.data
    HL = [] if v is None else v.data
    HH = [] if d is None else d.data
    return LL, LH, HL, HH


def getEblock(LH, HL, HH):
    sumEnergy = 0.0
    nB = 0.0
    # 𝐸(𝑥, 𝑦) = (𝐻𝐿 (𝑥, 𝑦)^2 + 𝐿𝐻(𝑥, 𝑦)^2 + 𝐻𝐻(𝑥, 𝑦)^2)
    for (h, v, d) in itertools.zip_longest(LH, HL, HH, fillvalue=[0, 0]):
        for (a, b, c) in itertools.zip_longest(h, v, d, fillvalue=0):
            sumEnergy += a*a + b*b + c*c
            nB = nB+1

    try:
        eB = 1/nB * sumEnergy
    except:
        eB = 0.0
    return eB
