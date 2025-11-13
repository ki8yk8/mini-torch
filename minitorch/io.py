import pickle as pkl

def save(state_dict, path):
	with open(path, "wb") as fp:
		pkl.dump(state_dict, fp)