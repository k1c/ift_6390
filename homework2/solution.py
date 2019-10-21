import numpy as np

class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

#TODO check edges cases for m (can we assume that the max label number is smaller than m?)
    def make_one_versus_all_labels(self, y, m):
        """
	y : numpy array of shape (n,)
	m : int (in this homework, m will be 10)
	returns : numpy array of shape (n,m)
	"""
        ova = np.full((y.shape[0], m), -1)
        for i, row in enumerate(ova):
            row[y[i]] = 1
        return ova

    def compute_loss(self, x, y):
        """
	x : numpy array of shape (minibatch size, 401)
	y : numpy array of shape (minibatch size, 10)
	returns : float
	"""
        scores = x.dot(self.w)
        margins = np.multiply(scores, y)
        loss = np.mean(np.sum(np.maximum(0, 1 - margins), axis=1))
        loss = self.C * loss
        # Computes de regularization term
        loss += 0.5 * np.sum(np.linalg.norm(self.w, ord=2, axis=0))
        return loss

    def compute_gradient(self, x, y):
        """
	x : numpy array of shape (minibatch size, 401)
	y : numpy array of shape (minibatch size, 10)
	returns : numpy array of shape (401, 10)
	"""
        scores = x.dot(self.w)
        # margin = <wj,(xi,yi)>
        margins = np.multiply(scores, y)
        # Indicator
        active = (margins < 1).astype(float)
        # plain gradients
        grad = np.dot(x.T, np.multiply(1 - margins,  -np.multiply(y, active)))
        # grad = 2 * C * grad / n
        grad = 2 * self.C * grad / y.shape[0]
        # add the regularization term
        grad += self.w
        return grad

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
	x : numpy array of shape (number of examples to infer, 401)
	returns : numpy array of shape (number of examples to infer, 10)
	"""
        y = x.dot(self.w)
        y_ova = -1 * np.ones(y.shape)
        y_ova[np.arange(y_ova.shape[0]), np.argmax(y, axis=1)] = 1
        return y_ova

    def compute_accuracy(self, y_inferred, y):
        y_ova[:, class_index] = 1
        return y_ova

    def compute_accuracy(self, y_inferred, y):
        """
	y_inferred : numpy array of shape (number of examples, 10)
	y : numpy array of shape (number of examples, 10)
	returns : float
	"""
        return np.sum(np.all(y == y_inferred, axis=1).astype(float)) / y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x, y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train, y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test, y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")


    Cs = [1, 3, 10, 30, 100, 300]
    etas = [0.01, 0.001, 0.0001]
    best_accuracy = 0.
    beat_C, beat_eta = None, None
    for C in Cs:
        for eta in etas:
            print(f"C {C}, eta {eta}")
            svm = SVM(eta=eta, C=C, niter=200, batch_size=5000, verbose=True)
            train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_C = C
                best_eta = eta
    print(f"Best results with C {best_C}, eta {eta}: test {tes_accuracy}, train {train_accuracy}")
    # to compute the gradient or loss before training, do the following:
    #y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    #svm.w = np.zeros([401, 10])
    #grad = svm.compute_gradient(x_train, y_train_ova)
    #loss = svm.compute_loss(x_train, y_train_ova)


    # to infer after training, do the following:
    #y_inferred = svm.infer(x_test)


