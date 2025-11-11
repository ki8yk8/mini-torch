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