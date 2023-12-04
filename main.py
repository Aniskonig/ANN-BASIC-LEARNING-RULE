import tkinter as tk
from tkinter import ttk, messagebox
from trainingNetwork import TrainingNetwork
import numpy as np

# GUI class for creating the graphical user interface
class GUI:
    def __init__(self, master):
        """
        Initialize the GUI with input fields, checkboxes, and dropdowns for configuring a neural network.

        Parameters:
        - master (tk.Tk): The root window of the GUI.
        """
        self.master = master
        self.master.title("Training network - ANN's HomeWork")

        # Create input fields, checkboxes, and variables for vectors and targeted values
        self.labels = []
        self.entries = []
        self.check_vars = []
        self.target_label = []
        self.target_entries = []
        self.target_check_vars = []

        label_data = tk.Label(master, text="Check and Enter the training data:")
        label_data.grid(row=0, column=0, sticky=tk.E)

        for i in range(1, 7):
            vector_check_var = tk.BooleanVar()
            vector_check_var.set(False)

            label = tk.Label(master, text=f"X {i}:")
            entry = tk.Entry(master, state=tk.DISABLED)
            checkbox = tk.Checkbutton(master, variable=vector_check_var,
                                      command=lambda j=i - 1, var=vector_check_var: self.toggle_entry(j, var))

            target_check_var = tk.BooleanVar()
            target_check_var.set(False)

            target_label = tk.Label(master, text=f"d {i}:")
            target_entry = tk.Entry(master, state=tk.DISABLED, width=7)
            target_checkbox = tk.Checkbutton(master, variable=target_check_var,
                                             command=lambda j=i - 1, var=target_check_var: self.toggle_target_entry(j,
                                                                                                                    var))

            self.labels.append(label)
            self.entries.append(entry)
            self.check_vars.append(vector_check_var)
            self.target_label.append(target_label)
            self.target_entries.append(target_entry)
            self.target_check_vars.append(target_check_var)

            label.grid(row=i, column=0, sticky=tk.E)
            entry.grid(row=i, column=1)
            checkbox.grid(row=i, column=2, sticky=tk.W)

            target_label.grid(row=i, column=3, sticky=tk.E)
            target_entry.grid(row=i, column=4)
            target_checkbox.grid(row=i, column=5, sticky=tk.W)

        # Create Learning rules dropdown list
        label_dropdown = tk.Label(master, text="Select a Learning rule:")
        values = ["Hebbian", "Perceptron", "Delta", "Widrow-Hoff"]  # Add your desired items
        self.learning_rule_dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(master, textvariable=self.learning_rule_dropdown_var, values=values)
        label_dropdown.grid(row=7, column=0)
        dropdown.grid(row=7, column=1)

        # Bind the function to the ComboboxSelected event
        dropdown.bind("<<ComboboxSelected>>", self.on_learning_rule_selected)

        # Create steps label and entry
        label_steps = tk.Label(master, text="Steps:")
        self.entry_steps = tk.Entry(master, width=7)
        label_steps.grid(row=7, column=3, sticky=tk.E)
        self.entry_steps.grid(row=7, column=4)

        # Create Activation Function dropdown list
        label_af_dropdown = tk.Label(master, text="Select an Activation function:")
        values_af = ["Bipolar Continuous", "Bipolar Binary", "Sng"]  # Add your desired items
        self.activation_function_dropdown_var = tk.StringVar()
        dropdown_af = ttk.Combobox(master, textvariable=self.activation_function_dropdown_var, values=values_af)
        label_af_dropdown.grid(row=8, column=0)
        dropdown_af.grid(row=8, column=1)

        # Bind the function to the ComboboxSelected event
        dropdown_af.bind("<<ComboboxSelected>>", self.on_activation_function_selected)

        # Create α label and entry
        label_alpha = tk.Label(master, text="α:")
        self.entry_alpha = tk.Entry(master, width=7)
        label_alpha.grid(row=8, column=3, sticky=tk.E)
        self.entry_alpha.grid(row=8, column=4)

        # Create initial weight label and entry
        label_init_w = tk.Label(master, text="Initial Weight:")
        self.entry_init_w = tk.Entry(master)
        label_init_w.grid(row=9, column=0, sticky=tk.E)
        self.entry_init_w.grid(row=9, column=1)

        # Create lambda label and entry
        label_lambda = tk.Label(master, text="λ:")
        self.entry_lambda = tk.Entry(master, width=7)
        label_lambda.grid(row=9, column=3, sticky=tk.E)
        self.entry_lambda.grid(row=9, column=4)

        space = tk.Label()
        space.grid()

        # Create a label to display the final weights
        self.label_final_weights = tk.Label(master, text="Final Weights:")
        self.label_final_weights.grid(row=11, column=0, sticky=tk.E)

        # Create a text widget to display the final weights
        self.text_final_weights = tk.Text(master, height=1, width=30, state=tk.DISABLED, background="black", fg="white")
        self.text_final_weights.grid(row=11, column=1, columnspan=4)

        space = tk.Label()
        space.grid()

        # Create submit button
        submit_button = tk.Button(master, text="Train", command=self.start_training, background="gold", width=15,
                                  height=1)
        submit_button.grid(row=13, columnspan=7)

        space = tk.Label()
        space.grid()

        # Create copyright label
        copyright_label = tk.Label(master, text="© 2023 Mohammed Anis Oukebdane, ANN course's homework.")
        copyright_label.grid(row=15, column=0, columnspan=7)

    def toggle_entry(self, index, check_var):
        """
        Enable/disable the vector input field based on the checkbox state.

        Parameters:
        - index (int): Index of the input field.
        - check_var (tk.BooleanVar): Variable associated with the checkbox.
        """
        state = tk.NORMAL if check_var.get() else tk.DISABLED
        self.entries[index].config(state=state)

    def toggle_target_entry(self, index, check_var):
        """
        Enable/disable the targeted value input field based on the checkbox state.

        Parameters:
        - index (int): Index of the targeted value input field.
        - check_var (tk.BooleanVar): Variable associated with the checkbox.
        """
        state = tk.NORMAL if check_var.get() else tk.DISABLED
        self.target_entries[index].config(state=state)

    def on_learning_rule_selected(self, event):
        """
        Handle the learning rule selection from the dropdown.

        Parameters:
        - event: The event triggered by the learning rule dropdown.
        """
        selected_item = self.learning_rule_dropdown_var.get()
        print("Selected Item:", selected_item)

    def on_activation_function_selected(self, event):
        """
        Handle the activation function selection from the dropdown.

        Parameters:
        - event: The event triggered by the activation function dropdown.
        """
        selected_item = self.activation_function_dropdown_var.get()
        print("Selected Item:", selected_item)

    def start_training(self):
        """
        Start the training process by retrieving user input, configuring the neural network, and displaying the results.
        """
        # Retrieve input values for vectors and their associated targeted values
        active_vectors = [entry.get() for entry, check_var in zip(self.entries, self.check_vars) if check_var.get()]
        targeted_values = [target_entry.get() if check_var.get() else None for target_entry, check_var in
                           zip(self.target_entries, self.target_check_vars)]
        initial_weight_str = self.entry_init_w.get().strip()
        alpha_str = self.entry_alpha.get().strip()
        steps_str = self.entry_steps.get().strip()
        selected_learning_rule = self.learning_rule_dropdown_var.get().strip()
        selected_activation_function = self.activation_function_dropdown_var.get().strip()

        # Check if initial weight and learning rate (α) are empty
        if not initial_weight_str:
            messagebox.showerror("Error", "Initial Weight cannot be empty")
            return

        if not alpha_str:
            messagebox.showerror("Error", "Learning Rate (α) cannot be empty")
            return

        if not steps_str:
            messagebox.showerror("Error", "Steps cannot be empty")
            return

        if not selected_learning_rule:
            messagebox.showerror("Error", "Choose a learning rule")
            return

        if not selected_activation_function:
            messagebox.showerror("Error", "Choose an activation function")
            return

        try:
            initial_weight = [float(i) for i in initial_weight_str.split(',')]
            initial_weight = np.array(initial_weight)

            processed_vectors, processed_targets = self.process_inputs(active_vectors, targeted_values)
            training_data = []

            print("processed_vectors:", processed_vectors)
            print("processed_targets:", processed_targets)

            for i, vector in enumerate(processed_vectors):
                if processed_targets:
                    training_data.append([vector, processed_targets[i]])
                else:
                    training_data.append(np.array(vector))

            print("training_data:", training_data)
            print("Initial Weight:", initial_weight)

            training_net = TrainingNetwork(num_inputs=6, initial_weights=initial_weight)

            # Set training data, learning rate, and number of training steps
            training_net.set_training_data(training_data)
            training_net.set_learning_rate(float(alpha_str))
            training_net.set_num_steps(int(self.entry_steps.get()))
            training_net.set_precision(4)
            training_net.set_field_parameter(float(self.entry_lambda.get()))

            # Set activation function
            activation_function_type = str(self.activation_function_dropdown_var.get())
            if activation_function_type == "Bipolar Continuous":
                training_net.set_activation_function(
                    lambda x: training_net.bipolar_continuous_function(x, float(self.entry_lambda.get())))
            elif activation_function_type == "Bipolar Binary":
                training_net.set_activation_function(lambda x: training_net.bipolar_binary_function(x))
            elif activation_function_type == "Sng":
                training_net.set_activation_function(lambda x: training_net.sgn_function(x))

            learning_rule_type = str(self.learning_rule_dropdown_var.get())

            if learning_rule_type == "Hebbian":
                training_net.set_learning_rule(training_net.hebbian_learning_rule)
            elif learning_rule_type == "Perceptron":
                training_net.set_learning_rule(training_net.perceptron_learning_rule)
            elif learning_rule_type == "Delta":
                training_net.set_learning_rule(training_net.delta_learning_rule)
            elif learning_rule_type == "Widrow-Hoff":
                training_net.set_learning_rule(training_net.widrowHoff_learning_rule)

            # Train the network and print the final weights
            final_weights = training_net.train()

            # Display the final weights in the text widget
            self.text_final_weights.config(state=tk.NORMAL)
            self.text_final_weights.delete(1.0, tk.END)
            self.text_final_weights.insert(tk.END, f"{final_weights}")
            self.text_final_weights.config(state=tk.DISABLED)
        except ValueError as e:
            print("Error", str(e))
            messagebox.showerror("Error", str(e))

    def process_inputs(self, vectors, targeted_values):
        """
        Process user inputs, splitting vectors and converting them to lists of floats.

        Parameters:
        - vectors (list): List of user-input vectors.
        - targeted_values (list): List of user-input targeted values.

        Returns:
        - processed_vectors (list): List of processed vectors.
        - processed_targets (list): List of processed targeted values.
        """
        processed_vectors = []
        processed_targets = []

        for vector, target_value in zip(vectors, targeted_values):
            components = vector.split(',')
            components = np.array([float(component.strip()) for component in components])
            processed_vectors.append(components)

            if target_value is not None:
                processed_targets.append(float(target_value.strip()))

        return processed_vectors, processed_targets


# Entry point of the program
def main():
    """
    Main function to initialize the Tkinter root window and instantiate the GUI.
    """
    root = tk.Tk()
    GUI(root)
    root.resizable(False, False)
    photo = tk.PhotoImage(file="logo.gif")
    root.iconphoto(False, photo)
    root.mainloop()

# Check if the script is being run as the main module
if __name__ == "__main__":
    main()
