# Omid55

def show_matrix(matrix, people_names=None):
    sns.set(rc={"figure.figsize": (8, 20)})
    sns.heatmap(matrix, cmap='Blues')#sns.cubehelix_palette(8))
    # seting xticks
    ax = plt.axes()
    ax.set_xticks(np.array(range(len(periods)))+0.5, minor=True)
    labels = [p[0]+' to '+p[1] for p in periods]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.array(range(len(pp.traders)))+0.5, minor=True)
    if people_names is None:
        people_names = sorted(pp.traders.keys(), reverse=True)
    ax.set_yticklabels(people_names, rotation=0)