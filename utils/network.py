import keras.backend as K

def euclidean_distance(a,b):
    return K.sqrt(K.sum(K.square((a-b)), axis=1))

def triplet_loss(y_true, anchor_positive_negative_tensor):
    anchor = anchor_positive_negative_tensor[:,:,0]
    positive = anchor_positive_negative_tensor[:,:,1]
    negative = anchor_positive_negative_tensor[:,:,2]

    Dp = euclidean_distance(anchor, positive)
    Dn = euclidean_distance(anchor, negative)

    return K.maximum(0.0, 1+K.mean(Dp-Dn))
