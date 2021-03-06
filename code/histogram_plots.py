from load_data import *


# subset dataframe to X
X = raw_data[numerical_attrs].values
C = len(np.unique(raw_data["body_style"]))
M = X.shape[1] # num attributes
N = X.shape[0] # num obs


# Plot histograms
fig = plt.figure(figsize=(16,10))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    plt.subplot(u,v,i+1)
    plt.hist(X[:,i], bins=14, color="dodgerblue")
    # plt.xlabel(numerical_attrs[i])
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: plt.yticks([])
    plt.title(numerical_attrs[i])


# plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle("Distributions of Values for Numerical Attributes", fontsize="xx-large")
fig.tight_layout()
# show or save plot
# plt.show()
plt.savefig("../out/plots/histograms.png", dpi=200)


# histogram of single attribute
fig = plt.figure()
plt.hist(X[:,num_attr2idx["peak_rpm"]], bins=50, color="dodgerblue")
plt.ylim(0, N)
plt.title("Peak RPM", fontsize="xx-large")
fig.tight_layout()
# show or save plot
# plt.show()
plt.savefig("../out/plots/histogram_peak_rpm.png", dpi=200)
