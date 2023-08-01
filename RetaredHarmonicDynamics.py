import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')


def animate_particle(N, x_data, y_data,
                     x_patch=None,
                     y_patch=None,
                     f=1, xlim=(-2, 2), ylim=(-2, 2)):
    with_patch = x_patch is not None and y_patch is not None
    fig = plt.figure(figsize=(5, 5))
    fig.set_dpi(100)
    ax = plt.axes(xlim=xlim, ylim=ylim)
    line, = ax.plot([], [], lw=2)
    if with_patch:
        patch = plt.Circle((0, 0), 0.1, fc='y')

    # initialization function 
    def init():
        # creating an empty plot/frame 
        line.set_data([], [])
        if with_patch:
            patch.center = (0, 0)
            ax.add_patch(patch)
            return line, patch
        return line,

        # animation function 

    def animate(i):
        frame = i * f
        line.set_data(x_data(frame), y_data(frame))
        if with_patch:
            patch.center = (x_patch(frame), y_patch(frame))
            return patch, line,
        return line,

    # call the animator	 
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=N // f, interval=20, blit=True)
    plt.close()
    return anim
