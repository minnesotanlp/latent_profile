import numpy as np

topic = 0
corr = 2

fname = f"experiments/core/2/{topic}/{corr}/conversations.npy"

conversations = np.load(fname)


for idx0 in range(conversations.shape[0]):
    for idx1 in range(conversations.shape[1]):
        for idx2 in range(conversations.shape[2]):
            for idx3 in range(conversations.shape[3]):
                conversations[idx0, idx1, idx2, idx3] = conversations[idx0, idx1, idx2, idx3].replace('Goodbye.', '')

fupdated = f"experiments/core/2/{topic}/{corr}/clean_conversations.npy"

np.save(fupdated, conversations)
# flattened = conversations.reshape(-1, conversations.shape[-1])



# rand_conv = np.random.randint(0, flattened.shape[0])

# while flattened[rand_conv][0] == '':
#     rand_conv = np.random.randint(0, flattened.shape[0])

# for res in flattened[rand_conv]:
#     print(res)
#     print('\n\n')