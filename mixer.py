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
        x = torch.exp(x)
        return x

# Define the path to the saved model
model_path = './models/simple_nn_model.pth'

# Load the model
model = SimpleNN(X_test_tensor.shape[1], y_test_tensor.shape[1])
model.load_state_dict(torch.load(model_path))
model.eval() 

print("Model loaded successfully.")


y_stds_tensor = torch.tensor(stds.loc[y_cols].to_numpy(), dtype=torch.float32)
y_means_tensor = torch.tensor(means.loc[y_cols].to_numpy(), dtype=torch.float32)
X_stds_tensor = torch.tensor(stds.drop(index=y_cols).to_numpy(), dtype=torch.float32)
X_means_tensor = torch.tensor(means.drop(index=y_cols).to_numpy(), dtype=torch.float32)

def denormalizeInputs(inputs):
    return inputs.view(-1, 1) * X_stds_tensor + X_means_tensor

def denormalizeOutputs(outputs):
    return outputs.view(-1, 1) * y_stds_tensor + y_means_tensor

    
material_costs_df = pd.read_csv("./data/material_costs_estimates.csv", index_col=0).drop(columns=["Category"])
material_costs_tensor = torch.tensor(material_costs_df.to_numpy(), dtype=torch.float32)

def materials_cost(inputs):
    return (inputs * material_costs_tensor).sum()

def objective_function(inputs, doPrint=False, min_strength = 60):
    
    outputs = denormalizeOutputs(model(inputs))
    strength = outputs[3, -1]
    cost = materials_cost(denormalizeInputs(inputs))

    if (doPrint):
        print("STRENGTH:", strength.item(), "COST:", cost.item())

    objective = torch.where(strength < min_strength, strength, -(cost ** 2) / 1.7e9                            )
    return objective


def optmizie(min_strength):

    SAMPLES = 1
    ITERATIONS = 1500
    MIN_STRENGTH = min_strength

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
            grads = torch.autograd.grad(objective_function(inputs, doPrint=doPrint, min_strength=MIN_STRENGTH), inputs, grad_outputs=torch.ones_like(objective_function(inputs)), retain_graph=True)[0]

            inputs = torch.clamp(inputs + grads * 0.01, min=-(X_means_tensor / (X_stds_tensor - 1e-8)).view(1, -1))

            # print("OUTPUT:", denormalizeOutputs(model(inputs)))
        
        cost = materials_cost(denormalizeInputs(inputs))
        if (cost < lowest_cost):
            lowest_cost = cost
            best_inputs = inputs
            print("NEW BEST:")
            objective_function(inputs, doPrint=True, min_strength=MIN_STRENGTH)
        


    print("RESULT:")
    X_cols = [col for col in test_df.columns if col not in y_cols]
    print("\nIngredients:\n", pd.DataFrame(denormalizeInputs(best_inputs).view(1, -1).detach().numpy(), columns=X_cols))
    print("\nPredicted results:\n", pd.DataFrame(denormalizeOutputs(model(best_inputs)).view(1, -1).detach().numpy(), columns=y_cols))
    print("Cost:", lowest_cost.item())


RUNNING = True
while RUNNING:
    try:
        min_strength = float(input("Enter the minimum strength: "))
        optmizie(min_strength)
    except ValueError as e:
        print("Please enter a valid number.")
        print(f"Error: {e}")
