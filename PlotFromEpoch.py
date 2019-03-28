import numpy as np
from plot_graph import plot_training_curve
print('Enter Model path (Model_Epoch#)')
model_path=input()
print('Enter Number of Epochs (# from modelpath+1)')
epochs=input()


np.savetxt("{}_train_err.csv".format(model_path), train_err)
np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
np.savetxt("{}_val_err.csv".format(model_path), val_err)
np.savetxt("{}_val_loss.csv".format(model_path), val_loss)


plot_training_curve('model_path',epochs)
