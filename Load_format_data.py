data = np.load('MNIST_CorrNoise.npz')

x_train = data['x_train']  #x_train is an array of image data with shape (num_samples, 28, 28).
y_train = data['y_train']  #y_train is an array of digit labels (integers in range 0-9) with shape.
num_cls = len(np.unique(y_train)) # this function will sort out the unique digit labels.
print('Number of classes: ' + str(num_cls))

print('Example of handwritten digit with correlated noise: \n')

k = 3000

plt.imshow(np.squeeze(x_train[k,:,:]))  #function .squeeze() will remove single-dimensional entries from the shape of an array.
plt.show()
print('Class: '+str(y_train[k])+'\n') #digit label at k=3000.

# RESHAPE and standarize
x_train = np.expand_dims(x_train/255,axis=3) #The function np.expand will expand the array by inserting new axis at the specified position <here the position is 3>.
# All images are grayscale images hence they have only one channel; thats why dimesion=1 in 28*28*1.
# Convert a class vector (integers) to binary class matrix.

y_train = to_categorical(y_train, num_cls)

print('Shape of x_train: '+str(x_train.shape))
print('x_train has '+str(x_train.shape[0]), 'train samples')

print('Shape of y_train: '+str(y_train.shape))
print('y_train has '+str(y_train.shape[0]), 'train samples')
