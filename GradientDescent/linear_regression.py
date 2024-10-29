import numpy as np
class LinearRegression():
    def __init__(self):
        self.w = 0.1
        self.b = 0.1

    def set(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x):
        pred = self.w * x + self.b
        return pred

    def obj_func_SE(self, x, y):
        ### MSE
        pred = self.predict(x)
        err = y -  pred
        SE = err * err
        return SE

    def grad_obj_func(self, x, y):
        grad_w = 0.0
        grad_b = 0.0
        pred = self.predict(x)
        ## d_obj/d_w = −2x(y−(wx+b))
        grad_w = -2 * x * (y-pred)
        ## d_obj/d_b = −2(y−(wx+b))
        grad_b = -2 * (y - pred)
        return np.array([grad_w, grad_b])

    def update_params(self, delta_w, delta_b):
        self.w = self.w + delta_w
        self.b = self.b + delta_b

    def print_param(self):
        print("Model param: w: %.2f, b: %.2f" % (self.w, self.b))
