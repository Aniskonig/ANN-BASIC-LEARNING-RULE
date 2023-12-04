import math
import numpy as np


class TrainingNetwork:
    def __init__(self, num_inputs, initial_weights=None):
        """
        Initialize a Hebbian Network.

        Parameters:
        - num_inputs (int): The number of input nodes in the network.
        - initial_weights (array, optional): Initial values for weights. If not provided, weights are initialized to zeros.
        """
        # Initialize the number of inputs
        self.num_inputs = num_inputs
        # Initialize weights to zeros or user-provided values
        self.weights = np.array(initial_weights) if initial_weights is not None else np.zeros(num_inputs)
        # Default activation function is sgn function
        self.activation_function = self.sgn_function
        # Default learning rule is Hebbian
        self.learning_rule = self.hebbian_learning_rule
        # Default decimal precision is set to all (-1)
        self.decimal_precision = -1
        # Default field parameter is 1
        self.field_parameter = 1

    def set_activation_function(self, activation_function):
        """
        Set the activation function for the network.

        Parameters:
        - activation_function (function): The activation function to be used.
        """
        self.activation_function = activation_function

    def set_learning_rule(self, learning_rule):
        """
        Set the learning rule for weight updates.

        Parameters:
        - learning_rule (function): The learning rule to be used for weight updates.
        """
        self.learning_rule = learning_rule

    def bipolar_continuous_function(self, u, field):
        """
        Return the value of the bipolar continuous activation function.

        Parameters:
        - u (float): The input multiplied by the transpose of the previous weight.
        - field (float): The field parameter for the activation function.

        Returns:
        - float: The output of the bipolar continuous activation function.
        """
        return (2 / (1 + math.exp(-(field * u)))) - 1

    def bipolar_binary_function(self, u, threshold=0):
        """
        Return the value of the bipolar binary activation function.

        Parameters:
        - u (float): The input multiplied by the transpose of the previous weight.
        - threshold (float): The threshold for the binary activation. Default is 0.

        Returns:
        - int: The output of the bipolar binary activation function (either -1 or 1).
        """
        return -1 if u <= threshold else 1

    def sgn_function(self, u, threshold=0):
        """
        Return the value of the sign function.

        Parameters:
        - u (float): The input multiplied by the transpose of the previous weight.
        - threshold (float): The threshold for the binary activation. Default is 0.

        Returns:
        - int: The output of the bipolar binary activation function (either -1 or 1).
        """
        return 1 if u > threshold else (-1 if u < threshold else 0)

    def set_training_data(self, training_data):
        """
        Set the training data for the network.

        Parameters:
        - training_data (list): List of input patterns for training.
        """
        self.training_data = training_data

    def set_learning_rate(self, learning_rate):
        """
        Set the learning rate for weight updates.

        Parameters:
        - learning_rate (float): The learning rate to be used for weight updates.
        """
        self.learning_rate = learning_rate

    def set_num_steps(self, num_steps):
        """
        Set the number of training steps.

        Parameters:
        - num_steps (int): The number of steps to train the network.
        """
        self.num_steps = num_steps

    def set_precision(self, decimal_precision):
        """
       Define the decimal precision for all used numbers

       Parameters:
       - decimal_precision (int): The desired decimal precision.
       """
        self.decimal_precision = decimal_precision

    def set_field_parameter(self, field_parameter):
        """
       Define the field parameter

       Parameters:
       - decimal_precision (int): The desired decimal precision.
       """
        self.field_parameter = field_parameter

    def adjust_number_precision(self, number):
        """
       Adjust a number to the decimal precision

       Parameters:
       - number (float): The number to be adjusted.

       Returns:
        - float: The adjusted number to the desired decimal precision.
       """
        if self.decimal_precision == -1:
            return number
        else:
            return round(number, self.decimal_precision)

    def hebbian_learning_rule(self):
        """
        Hebbian learning rule for weight updates.

        Returns:
        - array: The weight updates based on the Hebbian learning rule.
        """
        # Training the network for the specified number of steps
        # for step in range(self.num_steps):
        # Iterate over each training example
        for input_data in self.training_data:
            print("weights", self.weights)
            print("input_data", input_data)

            u = self.adjust_number_precision(np.dot(input_data, self.weights))
            print("u", u)
            predicted_output = self.adjust_number_precision(self.activation_function(u))
            print("f(u)", predicted_output)
            delta_weights = self.learning_rate * input_data * predicted_output
            print("W + delta_weights", self.weights + delta_weights)
            # Update weights using the specified learning rule
            # delta_weights = self.learning_rule(input_data)
            self.weights = self.weights + delta_weights

        # Return the final weights after training
        return self.weights

    def perceptron_learning_rule(self):
        """
        Perceptron learning rule for weight updates.

        Returns:
        - array: The weight updates based on the perceptron learning rule.
        """

        # Training the network for the specified number of steps
        # for step in range(self.num_steps):
        #     print("step------------------", step + 1)

        counter = 0
        # Iterate over each training example
        for i, (input_data, target_output) in enumerate(self.training_data):
            if counter < self.num_steps:
                u = self.adjust_number_precision(np.dot(input_data, self.weights))
                predicted_output = self.adjust_number_precision(self.activation_function(u))
                print("weights", self.weights)
                print(f"X {i+1}")
                print("input_data", input_data)
                print("target_output", target_output)
                print("predicted_output", predicted_output)
                # Check if predicted output matches the target output
                if predicted_output != target_output:
                    error = target_output - predicted_output
                    print("predicted_output != target_output")
                    delta_weights = self.learning_rate * int(error) * input_data
                    print("delta_weights", delta_weights)
                    self.weights = self.weights + delta_weights
                else:
                    # If it's the last input, return the weights
                    if i == len(self.training_data) - 1:
                        return self.weights  # Stop training if it's the last input
                counter += 1
        return self.weights  # the last input if the steps are less than the needed

    def delta_learning_rule(self):
        """
        Delta learning rule for weight updates.

        Returns:
        - array: The weight updates based on the perceptron learning rule.
        """
        # Training the network for the specified number of steps
        # entry_steps = self.num_steps / 2
        # steps = int(entry_steps) + 1 if self.num_steps % 2 != 0 else int(entry_steps)
        counter = 0
        #
        # for step in range(steps):
        #     print("step------------------", step + 1)
        # Iterate over each training example
        for i, (input_data, target_output) in enumerate(self.training_data):
            if counter < len(self.training_data):
                print("step ------------------", counter + 1)
                u = self.adjust_number_precision(np.dot(input_data, self.weights))
                predicted_output = self.adjust_number_precision(self.activation_function(u))
                f_prime_u = self.adjust_number_precision(0.5*(1 - predicted_output ** 2))
                print("weights", self.weights)
                print("input_data", input_data)
                print("target_output", target_output)
                print("u", u)
                print("predicted_output", predicted_output)
                print("f_prime_u", f_prime_u)

                delta_weights = self.adjust_number_precision(self.learning_rate * (target_output - predicted_output)* f_prime_u) * input_data
                print("delta_weights", delta_weights)
                self.weights = self.weights + delta_weights
                counter += 1
            else:
                break

        return self.weights  # the last input if the steps are less than the needed

    def widrowHoff_learning_rule(self):
        """
        Widrow-Hoff learning rule for weight updates.

        Returns:
        - array: The weight updates based on the perceptron learning rule.
        """
        # Training the network for the specified number of steps
        # for step in range(self.num_steps):
        #     print("step------------------", step + 1)
        # Iterate over each training example
        counter = 0
        for i, (input_data, target_output) in enumerate(self.training_data):
            if counter < self.num_steps:
                print("step ------------------", counter + 1)
                u = self.adjust_number_precision(np.dot(input_data, self.weights))
                predicted_output = u
                f_prime_u = 1
                print("weights", self.weights)
                print("input_data", input_data)
                print("target_output", target_output)
                print("u", u)
                print("predicted_output", predicted_output)
                print("f_prime_u", f_prime_u)

                delta_weights = self.adjust_number_precision(self.learning_rate * (target_output - predicted_output)* f_prime_u) * input_data
                print("delta_weights", delta_weights)
                self.weights = self.weights + delta_weights
                counter += 1
            else:
                break

        return self.weights  # the last input if the steps are less than the needed

    def train(self):
        """
        Train the Hebbian network using the specified parameters.

        Returns:
        - array: The final weights of the network after training.
        """
        return self.learning_rule()
