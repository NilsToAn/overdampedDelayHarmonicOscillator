# %%
class test_calss:
    def run(self, *args):
        print("running")
        return self.main(*args)

    def my_name(self):
        print(self.__class__.__name__)


# %%
def blablablac(arg1, arg2):
    return arg1 + arg2


class my_test_class(test_calss):
    # def run(self):
    #    print("bla")

    def special(self):
        print("I am spezial")

    def main(self, *args):
        print(args)
        return blablablac(*args)


# %%
mytest = my_test_class()
# %%
mytest.run(4, 5)
# %%
mytest.special()
# %%
mytest.my_name()
# %%
gloabltest_class = test_calss()
# %%
gloabltest_class.my_name()


# %%
from functions import SimulationManager

my_sim_manager = SimulationManager()

N_p = 10_000
N_loop = 20
N_t = 1000
N_x = 21
ntau = 40
s = 1
dt = 1.5 / 40
x_0 = 0
force = "linear"
hist_sigma = 3

res = my_sim_manager.run(
    N_p=N_p,
    N_loop=N_loop,
    N_t=N_t,
    N_x=N_x,
    ntau=ntau,
    s=s,
    dt=dt,
    x_0=x_0,
    force=force,
    hist_sigma=hist_sigma,
)

res.keys()

# %%
res["x_s"]
# %%
from functions import get_eq_times

get_eq_times(0.5, 1 / 2, 0.75)
# %%
