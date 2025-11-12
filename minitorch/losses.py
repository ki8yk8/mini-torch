from .activations import LogSoftmax

class Loss:
	def __init__(self):
		pass

	def sanitize(self, output, actual, type="regression"):
		# if list all the output and actual should be list
		# and the length of list should be equal
		if isinstance(output, list):
			if not isinstance(actual, list):
				raise Exception("Output and Actual must be of type list")
			if not len(output) == len(actual):
				raise("Length of output, and actual must be equal")
		else:
			output, actual = [output], [actual]

		return output, actual

class MSE(Loss):
	def __init__(self):
		pass

	def __call__(self, output, actual):
		output, actual = self.sanitize(output, actual, type="regression")
		squared_error = [(o-a)**2 for o, a in zip(output, actual)]
		
		return sum(squared_error)/len(squared_error)

# difference between NLL and CE is that NLL assumes the input is logsoftmax
class NLL(Loss):
	def __init__(self):
		pass

	def __call__(self, output, target):
		if target > len(output) - 1:
			raise Exception(f"The target {target} is out of bound")

		return -output[target]

class CrossEntropy(Loss):
	def __init__(self):
		self.log_softmax = LogSoftmax(stability=True)

	def __call__(self, output, target):
		if target > len(output) -1:
			raise Exception(f"The target {target} is out of bound of output")

		log_probs = self.log_softmax(output)
		
		return NLL()(log_probs, target)


