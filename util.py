def split_data(data, labels, proportion):

    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    data = (data//255)
    return data[s[:split_idx]],
           data[s[split_idx:]],
           labels[s[:split_idx]],
           labels[s[split_idx:]]
