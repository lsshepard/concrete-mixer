from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import pandas as pd
import os
from mixer import optmizie, denormalizeInputs, denormalizeOutputs, model, test_df, y_cols, materials_cost, CO2_emissions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        
        # Extract parameters from the request
        samples = data.get('samples', 15)
        iterations = data.get('iterations', 1500)
        optimization_target = data.get('optimization_target', 'COST')
        input_constraints = data.get('input_constraints', [])
        output_constraints = data.get('output_constraints', [])
        
        # Debug logging
        print(f"Received request with parameters:")
        print(f"- samples: {samples}")
        print(f"- iterations: {iterations}")
        print(f"- optimization_target: {optimization_target}")
        print(f"- input_constraints: {input_constraints}")
        print(f"- output_constraints: {output_constraints}")
        
        # Call the optimizer function
        best_inputs = optmizie(
            SAMPLES=samples,
            ITERATIONS=iterations,
            OPTIMIZATION_TARGET=optimization_target,
            INPUT_CONSTRAINTS=input_constraints,
            OUTPUT_CONSTRAINTS=output_constraints
        )
        
        # Get column names for the ingredients
        X_cols = [col for col in test_df.columns if col not in y_cols]
        
        # Get the denormalized inputs for display
        denormalized_inputs = denormalizeInputs(best_inputs)
        
        # Convert to numpy for the ingredients display
        denormalized_inputs_np = denormalized_inputs.view(1, -1).detach().numpy()[0]
        ingredients = {X_cols[i]: float(denormalized_inputs_np[i]) for i in range(len(X_cols))}
        
        # Get the predicted results
        predicted_results = denormalizeOutputs(model(best_inputs)).view(1, -1).detach().numpy()[0]
        results = {y_cols[i]: float(predicted_results[i]) for i in range(len(y_cols))}
        
        # Calculate cost - matching exactly how it's done in the optimizer
        # Looking at line 183: cost = materials_cost(denormalizeInputs(inputs))
        cost = float(materials_cost(denormalizeInputs(best_inputs)).item())
        
        # Calculate CO2 - matching exactly how it's used in the objective function
        # This function internally calls denormalizeInputs, so we pass best_inputs
        co2 = float(CO2_emissions(best_inputs).item())
        
        return jsonify({
            'success': True,
            'ingredients': ingredients,
            'predicted_results': results,
            'cost': cost,
            'co2_emissions': co2
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in optimization: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_details
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Production settings
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Use Gunicorn for production
    from gunicorn.app.base import BaseApplication
    
    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
        
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)
        
        def load(self):
            return self.application
    
    options = {
        'bind': '0.0.0.0:3000',
        'workers': 4,
        'timeout': 120,
        'worker_class': 'sync'
    }
    
    StandaloneApplication(app, options).run() 