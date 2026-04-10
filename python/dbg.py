import matplotlib.pyplot as plt
import numpy as np
PATH = "../dbg/"


labels = ["v_h", "v_e", "v_r"]

values = np.load(PATH + "avg_vel.npy")
for vs in values.T:
    plt.plot(range(values.shape[0]), vs);
plt.legend(labels)
plt.title("average velocities")
plt.show()

values = np.load(PATH + "max_vel.npy")
for vs in values.T:
    plt.plot(range(values.shape[0]), vs);
plt.legend(labels)
plt.title("maximum velocities")
plt.show()

# plt.savefig(PATH + "max_vel.svg")

# values = np.load(PATH + "angle_convergence.npy")
#
# plt.plot(values)
# plt.title("angle change degrees")
# plt.ylim(-1.0, 1.0)
# plt.savefig(PATH + "e_dipoles_angle_convergence.svg")
#
# values = np.load(PATH + "norm_convergence.npy")
#
# plt.title("norm change")
# plt.plot(values/values[-1])
# plt.ylim(0.95, 1.05)
# plt.savefig(PATH + "e_dipoles_norm_convergence.svg")

