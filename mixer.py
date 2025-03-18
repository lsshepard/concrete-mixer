import torch
import pandas as pd
import torch.nn as nn

test_df = pd.read_csv('./data/test_df.csv', index_col=0)
means = pd.read_csv('./data/means.csv', index_col = 0)
stds = pd.read_csv('./data/stds.csv', index_col = 0)

y_cols = ["Målt konsistens [mm]", "Målt densitet [kg/m3]", "Målt luftinnhold [%]", "Sylinder 28 døgn"]

X_test_tensor = torch.tensor(test_df.drop(columns=y_cols).values, dtype=torch.float32)
y_test_tensor = torch.tensor(test_df[y_cols].values, dtype=torch.float32)

print("test data loaded successfully")


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the path to the saved model
model_path = './models/simple_nn_model.pth'

# Load the model
model = SimpleNN(X_test_tensor.shape[1], y_test_tensor.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval() 

print("Model loaded successfully.")


y_stds_tensor = torch.tensor(stds.loc[y_cols].to_numpy(), dtype=torch.float32).view(-1, 1)
y_means_tensor = torch.tensor(means.loc[y_cols].to_numpy(), dtype=torch.float32).view(-1, 1)
X_stds_tensor = torch.tensor(stds.drop(index=y_cols).to_numpy(), dtype=torch.float32).view(-1, 1)
X_means_tensor = torch.tensor(means.drop(index=y_cols).to_numpy(), dtype=torch.float32).view(-1, 1)

def denormalizeInputs(inputs):
    return inputs.view(-1, 1) * X_stds_tensor + X_means_tensor

def denormalizeOutputs(outputs):
    return torch.exp(outputs.view(-1, 1) * y_stds_tensor + y_means_tensor)

    
material_costs_df = pd.read_csv("./data/material_costs_estimates.csv", index_col=0).drop(columns=["Category"])
material_costs_tensor = torch.tensor(material_costs_df.to_numpy(), dtype=torch.float32)

def materials_cost(inputs):
    return (inputs * material_costs_tensor).sum()

def CO2_emissions(inputs):
    denormalizedInputs = denormalizeInputs(inputs)
    cement = denormalizedInputs[45, 0] + denormalizedInputs[50, 0] * 0.85
    CO2 = cement * 800
    return CO2

def objective_function(inputs, constraints, optimizationTarget="COST", doPrint=False):
    
    outputs = denormalizeOutputs(model(inputs))
    consistency = outputs[0, -1]
    density = outputs[1, -1]
    air = outputs[2, -1]
    strength = outputs[3, -1]
    denormalizedInputs = denormalizeInputs(inputs)
    cost = materials_cost(denormalizedInputs)
    cement = denormalizedInputs[45, 0] + denormalizedInputs[50, 0] * 0.85

    if (doPrint):
        print("CONSISTENCY:", consistency.item(), "DENSITY:", density.item(), "AIR:", air.item(), "STRENGTH:", strength.item(), "COST:", cost.item(), "CEMENT:", cement.item())

    attributes_map = {
        "consistency": 0,
        "density": 1,
        "air": 2,
        "strength": 3
    }
    
    objective = torch.tensor(0.0, requires_grad=True)

    for constraint in constraints:
        attribute_value = outputs[attributes_map[constraint["attribute"]], -1]
        # print("ATTR VAL: ", attribute_value)
        if (constraint["type"] == "min"):
            objective = objective + torch.where(attribute_value < constraint["value"], attribute_value - constraint["value"], torch.tensor(0.0))
        if (constraint["type"] == "max"):
            objective = objective + torch.where(attribute_value > constraint["value"], constraint["value"] - attribute_value, torch.tensor(0.0))

    # If all constraints are satisfied (objective is zero)
    if torch.all(objective == 0) or objective.item() == 0:
        
        if (optimizationTarget == "COST"):
            objective = -torch.pow(cost, 2) / torch.tensor(1.7e9, dtype=cost.dtype)
        elif (optimizationTarget == "CO2"):
            objective = -torch.pow(cement, 2)

    # objective = torch.where(strength < min_strength, strength, -(cost ** 2) / 1.7e9                            )
    return objective


def optmizie(SAMPLES=15, ITERATIONS=1500, OPTIMIZATION_TARGET="COST", INPUT_CONSTRAINTS=[], OUTPUT_CONSTRAINTS=[]):

    # SAMPLES = 15
    # ITERATIONS = 1500
    # OPTIMIZATION_TARGET = "COST"

    # OUTPUT_CONSTRAINTS = [
    #     {
    #     "attribute": "strength",
    #     "type": "min",
    #     "value": MIN_STRENGTH
    #     },
    #     {
    #     "attribute": "density",
    #     "type": "min",
    #     "value": 2500
    #     },
    #     {
    #     "attribute": "air",
    #     "type": "max",
    #     "value": 2.5
    #     }]

    # INPUT_CONSTRAINTS = [
    #     {
    #     "attribute_index": 50,
    #     "type": "min",
    #     "value": 400
    #     }
    # ]

    print("INPUT CONSTRAINTS:", INPUT_CONSTRAINTS)
    print("OUTPUT CONSTRAINTS:", OUTPUT_CONSTRAINTS)
    print("OPTIMIZATION TARGET:", OPTIMIZATION_TARGET)
    print("SAMPLES:", SAMPLES)
    print("ITERATIONS:", ITERATIONS)

    min_clamp = -(X_means_tensor / (X_stds_tensor - 1e-10)).view(1, -1)
    # Initialize max_clamp with very large values
    max_clamp = torch.full_like(min_clamp, float('inf'))

    for constraint in INPUT_CONSTRAINTS:
        # The normalization formula is the same for both min and max constraints
        normalized_value = (constraint["value"] - X_means_tensor[constraint["attribute_index"]]) / X_stds_tensor[constraint["attribute_index"]]
        
        if constraint["type"] == "min":
            # For min constraints, we set the minimum allowed normalized value
            min_clamp[0, constraint["attribute_index"]] = normalized_value
        elif constraint["type"] == "max":
            # For max constraints, we set the maximum allowed normalized value
            max_clamp[0, constraint["attribute_index"]] = normalized_value

    CONSTRAINTS = OUTPUT_CONSTRAINTS
    
    print("Optimizing different mixes...")

    best_inputs = None
    lowest_cost = float('inf')

    for i in range(SAMPLES):

        inputs = torch.normal(mean=0, std=1, size=(1, X_test_tensor.shape[1]), requires_grad=True)


        for j in range(ITERATIONS):

            outputs = model(inputs)

            # Compute gradients without backward()
            # doPrint = j == ITERATIONS -1
            doPrint = False
            grads = torch.autograd.grad(objective_function(inputs, CONSTRAINTS, OPTIMIZATION_TARGET, doPrint=doPrint), inputs, grad_outputs=torch.ones_like(objective_function(inputs, {})), retain_graph=True)[0]

            inputs = torch.clamp(inputs + grads * 0.01, min=min_clamp, max=max_clamp)

            # print("OUTPUT:", denormalizeOutputs(model(inputs)))
        
        cost = materials_cost(denormalizeInputs(inputs))
        if (cost < lowest_cost):
            lowest_cost = cost
            best_inputs = inputs
            print("NEW BEST:")
            objective_function(inputs, CONSTRAINTS, doPrint=True)
        


    print("RESULT:")
    X_cols = [col for col in test_df.columns if col not in y_cols]
    print("\nIngredients:\n", pd.DataFrame(denormalizeInputs(best_inputs).view(1, -1).detach().numpy(), columns=X_cols))
    print("\nPredicted results:\n", pd.DataFrame(denormalizeOutputs(model(best_inputs)).view(1, -1).detach().numpy(), columns=y_cols))
    print("Cost:", lowest_cost.item())
    
    return best_inputs


# if (__name__ == "__main__"):
#     RUNNING = True
#     while RUNNING:
#         try:
#             min_strength = float(input("Enter the minimum strength: "))
#             optmizie(min_strength)
#         except ValueError as e:
#             print("Please enter a valid number.")
#             print(f"Error: {e}")
