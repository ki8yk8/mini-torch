from .activations import LogSoftmax

class Loss:
	def __init__(self):
		pass

	def sanitize(self, output, actual):
		# if list all the output and actual should be list
		# and the length of list should be equal
		if isinstance(output, list):
			if not isinstance(actual, list):
				raise("Output and Actual must be of type list")
			if not len(output) == len(actual):
				raise("Length of output, and actual must be equal")
		else:
			output, actual = [output], [actual]

		return output, actual

class MSE(Loss):
	def __init__(self):
		pass

	def __call__(self, output, actual):
		output, actual = self.sanitize(output, actual)
		squared_error = [(o-a)**2 for o, a in zip(output, actual)]
		
		return sum(squared_error)/len(squared_error)

class NLL(Loss):
	def __init__(self):
		pass

	def __call__(self, output, actual):
		output, actual = self.sanitize(output, actual)

		# NLL = -log(p_y)
		correct_index = actual.index(1)
				
		if not correct_index:
			raise("There should be one positive target per sample")
		
		return -output[correct_index]

class CrossEntropy(Loss):
	def __init__(self):
		self.log_softmax = LogSoftmax(stability=True)

	def __call__(self, output, actual):
		output, actual = self.sanitize(output, actual)
		log_probs = self.log_softmax(output)
		
		return NLL()(log_probs, actual)


