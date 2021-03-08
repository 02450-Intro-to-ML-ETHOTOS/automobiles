from load_data import *

scope = "all"
y_measure = "pct"

# summmary statistics for categorical values
sumstats_categorical_attrs = {a: raw_data[a].value_counts().to_frame() for a in categorical_attrs}

cat_attrs = ["engine_location", "fuel_type", "make", "num_of_doors"] if scope == "selected" else categorical_attrs
figsize = (10,4) if scope == "selected" else (12,10)

fig = plt.figure(figsize=figsize)
M = len(cat_attrs)

if (len(cat_attrs) > 4):
    u = int(np.floor(np.sqrt(M))); v = int(np.ceil(float(M)/u))
else:
    u = 1; v = 4

N = raw_data.shape[0]

attrs = iter(cat_attrs)

for i in range(M):
    # use generator to iterate through categorical attributes
    # this is an alternative to calculating the index of the list, 
    # which will break, when the list is too short
    try:
        a = next(attrs)
    except StopIteration:
        break # out of for loop
    
    # calculate pct
    df = raw_data[a].value_counts().to_frame()
    df.rename(columns={a: "count"}, inplace=True)
    df["pct"] = df / N
    
    # plot
    plt.subplot(u,v,i+1)
    plt.bar(x=df.index, height=df[y_measure].values, width=0.8, color="dodgerblue")
    plt.title(a)

    # style xaxis
    fsize = 7 if a == "make" else 10
    rot = 90 if a == "make" else 45
    plt.xticks(fontsize=fsize, rotation=rot)

    # style yaxis
    if y_measure == "pct":
        plt.ylim(0,1) # if plotting pct, it's nice to limit y range to [0,1]
    
    plt.ylabel(y_measure)


plt.suptitle("Categorical Attribute Value Distributions", fontsize="xx-large")
fig.tight_layout()
# show or save plot
# plt.show()
plt.savefig(f"../out/plots/barplots_{y_measure}_{scope}.png", dpi=200)
