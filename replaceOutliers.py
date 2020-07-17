from sklearn import linear_model
import numpy as np
def replaceOutliers(x,y, threshold):
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1,1)
    ransac = linear_model.RANSACRegressor(residual_threshold=threshold)
    ransac.fit(x, y)
    # Predict data of estimated models
    line_y_ransac = ransac.predict(x)
    return line_y_ransac