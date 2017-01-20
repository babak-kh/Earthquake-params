import numpy as np



def interpolating(inputpath, outputpath, n=20, f1=0.4, f2=15):

    a1 = (np.log(f2)-np.log(f1))/(n-1)
    w1 = np.arange(0, n, dtype=float)
    w1[0] = f1

    for i in range(1, int(n)):
        w1[i] = np.around(np.exp(np.log(w1[0])+(i*a1)), decimals=4)

# ----------reading and interpolating spectrum and ratio matrices-----------

    ratio = np.genfromtxt(inputpath +
                          '/ratio.csv',delimiter=',')
    nratio = np.shape(ratio)

    spec = np.genfromtxt(inputpath +
                         '/spectrums.csv', delimiter=',')
    nspec = np.shape(spec)

    w2 = np.genfromtxt(inputpath + '/w2.csv')
    w2 = np.around(w2, decimals=4)

    ratiointerp = np.zeros((n+3, nratio[1]))
    specinterp = np.zeros((n+3, nspec[1]))

    ratiointerp[0:3, :] = ratio[0:3, :]
    specinterp[0:3, :] = spec[0:3, :]


    for i in range(0, nspec[1]):
        ratiointerp[3:, i] = np.around(np.interp(w1, w2, ratio[3:nratio[0], i]), decimals=4)
        specinterp[3:, i] = np.around(np.interp(w1, w2, spec[3:nratio[0], i]), decimals=4)

    # --------- saving data into results folder -------------------

    np.savetxt(outputpath + 'Interpolating/freqs.txt', w1,
               fmt='%0.4f', delimiter=',')
    np.savetxt(outputpath + 'Interpolating/specinterp.txt', specinterp,
               fmt='%.4e', delimiter=',')
    np.savetxt(outputpath + 'Interpolating/ratiointerp.txt', ratiointerp,
               fmt='%.4e', delimiter=',')
    string = 'Interpolated ratio matrix has the shape : %d %d \n' \
             'Interpolated spectrum matrix has the shape : %d %d' % (nratio[0], nratio[1], nspec[0], nspec[1])
    with open(outputpath + 'Interpolating/Test.txt', "w") as text_file:
        text_file.write(string)

    return w1, specinterp, ratiointerp

# ---------------testing--------------------

# w1, spec, rat = interpolating('/home/babak/PycharmProjects/Simulation/GeneralInversion/Inputs')
# np.savetxt('/home/babak/PycharmProjects/Simulation/GeneralInversion/Results/spec.txt', spec,
#                fmt='%0.2f', delimiter=',')
# np.savetxt('/home/babak/PycharmProjects/Simulation/GeneralInversion/Results/rat.txt', rat,
#                fmt='%0.2f', delimiter=',')
# np.savetxt('/home/babak/PycharmProjects/Simulation/GeneralInversion/Results/w1.txt', w1,
#                fmt='%0.2f', delimiter=',')