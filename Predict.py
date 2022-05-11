import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D,Lambda, MaxPooling2D, Input,Conv2DTranspose
import h5py
from matplotlib import pyplot as plt

#################
TrainFromScratch=False
#################

ModelName='Model.h5'
TestData='TestData.h5'
TrainData='TrainData.h5'
#################

def autoencoder(INPUT_SHAPE,OUTPUT_SHAPE):
    depth=6
    flt=2**(np.arange(depth)+4)+10
    inputs = Input(INPUT_SHAPE[1:])
    s = Lambda(lambda x: x ) (inputs)
    # Encoder
    x=s
    for I in range(depth):    
        x = Conv2D(flt[I], (3, 3), activation="selu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = BatchNormalization()(x)       
    # Decoder
    for I in range(depth):
        x = Conv2DTranspose(flt[depth-I-1], (3, 3), strides=2, activation="selu", padding="same")(x)
        x = BatchNormalization()(x)
    x = Conv2D(OUTPUT_SHAPE[-1], (3, 3), activation="sigmoid", padding="same")(x)
    
    outputs = Conv2D(OUTPUT_SHAPE[-1], (1, 1), activation='sigmoid') (x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse','accuracy']) 
    model.summary(line_length=128)
    return model
def readh5slice(FileName,FieldName,Slices):
    with h5py.File(FileName, "r") as f:
         A=f[FieldName][np.sort(Slices),...]
    return A
def h5size(Name,Field):
    import h5py
    with h5py.File(Name,'r') as f:
        Shape=f[Field].shape  
    return Shape

# Reloading the model
INPUT_SHAPE=h5size(TrainData,'X')
OUTPUT_SHAPE=h5size(TrainData,'X')
model=autoencoder(INPUT_SHAPE,OUTPUT_SHAPE)
model.load_weights(ModelName)

# Reading and predicting test data
INPUT_SHAPE=h5size(TestData,'X')
A=readh5slice(TestData,'X',np.arange(20))
B=readh5slice(TestData,'X',np.arange(20))
X=np.float32(A/255)
Y=np.float32(B/255)
Y2=model.predict(X)
acc=model.evaluate(X,Y)

# Plotting predictions
SA_slice=2  # change the SA_slice value from 0 to 9 to access different short axis slices of the heart
for samp in range(0,7):
    plt.imshow(np.concatenate((Y[samp,:,:,SA_slice],Y2[samp,:,:,SA_slice]),axis=1))
    plt.show()


